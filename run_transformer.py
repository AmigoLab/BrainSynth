#!/usr/bin/env python3
import os
from typing import Union, Tuple

import deepspeed
import numpy as np
import torch
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from fire import Fire
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.handlers.checkpoint import Checkpoint
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.utils import CommonKeys
from monai.handlers import CheckpointSaver, LrScheduleHandler, CheckpointLoader
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from src.handlers.general import (
    EvaluationHandler,
    MaxEpochsHandler,
    LoggingPreparationHandler,
    TensorBoardHandler,
    LossSummaryHandler,
    NpySaver,
    ConsolidateZeROHandler,
)
from src.inferer.transformer import (
    TransformerTrainingInferer,
    TransformerInferenceInferer,
)
from src.losses.transformer.transformer import CELoss
from src.metrics.transformer import CE
from src.networks.transformers.configure import get_transformer_network
from src.networks.transformers.img2seq_ordering import (
    Ordering,
    OrderingType,
    OrderingTransformations,
    RelativeSpatialPositioning,
)
from src.networks.vqvae.configure import get_vqvae_network
from src.utils.constants import (
    TransformerModes,
    TransformerConditioningType,
    TransformerVQVAEConditioningTypes,
)
from src.utils.general import get_gamma, basic_initialization, log_network_size
from src.utils.transformer import get_data_flow, prepare_batch, prepare_inference_batch


def training(config: dict):
    logger, device = basic_initialization(
        config=config, logger_name="Transformer-Training"
    )

    if config["conditionings"]:
        config["use_continuous_conditioning"] = (
            list(config["use_continuous_conditioning"])
            if type(config["use_continuous_conditioning"]) is tuple
            else [config["use_continuous_conditioning"]] * len(config["conditionings"])
        )
    else:
        config["use_continuous_conditioning"] = []

    training_loader, evaluation_loader, training_evaluation_loader = get_data_flow(
        config=config, logger=logger
    )

    if config["vqvae_checkpoint"]:
        vqvae_network = get_vqvae_network(
            config={
                "network": config["vqvae_network"],
                "use_subpixel_conv": config["vqvae_net_use_subpixel_conv"],
                "use_slim_residual": config["vqvae_net_use_slim_residual"],
                "no_levels": config["vqvae_net_no_levels"],
                "downsample_parameters": config["vqvae_net_downsample_parameters"],
                "upsample_parameters": config["vqvae_net_upsample_parameters"],
                "no_res_layers": config["vqvae_net_no_res_layers"],
                "no_channels": config["vqvae_net_no_channels"],
                "codebook_type": config["vqvae_net_codebook_type"],
                "num_embeddings": config["vqvae_net_num_embeddings"],
                "embedding_dim": config["vqvae_net_embedding_dim"],
                "embedding_init": config["vqvae_net_embedding_init"],
                "commitment_cost": config["vqvae_net_commitment_cost"],
                "decay": config["vqvae_net_decay"],
                "norm": config["vqvae_net_norm"],
                "dropout": config["vqvae_net_dropout"],
                "act": config["vqvae_net_act"],
                "output_act": config["vqvae_net_output_act"],
                "data_spatial_dimension": config["vqvae_net_data_spatial_dimension"],
                "data_num_channels": config["vqvae_net_data_num_channels"],
            }
        )

        vqvae_network.to(device)
        Checkpoint.load_objects(
            to_load={"network": vqvae_network},
            checkpoint=torch.load(config["vqvae_checkpoint"], map_location=device),
            strict=True,
        )

    else:
        vqvae_network = None

    config["vqvae_network"] = vqvae_network

    temp_batch = next(iter(training_loader))

    if config["vqvae_checkpoint"]:
        config["spatial_shape"] = (
            config["vqvae_network"]
            .index_quantize(temp_batch["quantization"].to(device))[
                config["vqvae_net_level"]
            ]
            .shape[1:]
        )

        if (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.BINARY.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = [2] * temp_batch[
                "quantization_binary_trace_dict"
            ]["total_count"].numpy().mean().astype(np.int)
        elif (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.CONTINUOUS.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = [-1] * temp_batch[
                "quantization_continuous_trace_dict"
            ]["total_count"].numpy().mean().astype(np.int)
        elif (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.NONE.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = []
        else:
            raise ValueError(
                f"VQVAE augmentation conditioning unknown. Received {config['vqvae_aug_conditionings']}, but valid "
                + f"options are {[e.value for e in TransformerVQVAEConditioningTypes]}"
            )
    else:
        config["spatial_shape"] = temp_batch["quantization"].shape[1:]
        config["vqvae_aug_conditioning_num_tokens"] = []

    config["ordering"] = Ordering(
        ordering_type=config["ordering_type"],
        spatial_dims=config["data_spatial_dimension"],
        dimensions=(1,) + config["spatial_shape"],
        reflected_spatial_dims=config["reflected_spatial_dims"],
        transpositions_axes=config["transpositions_axes"],
        rot90_axes=config["rot90_axes"],
        transformation_order=config["transformation_order"],
    )

    if config["data_spatial_dimension"] == 3:
        config["relative_spatial_pos_attr"] = RelativeSpatialPositioning(
            spatial_dims=config["data_spatial_dimension"],
            dimensions=(1,) + config["spatial_shape"],
            ordering=config["ordering"].get_sequence_ordering(),
            bucket_values=config["bucket_values"],
            bucket_beta=config["spatial_bias_max_dist"],
            conditioning_length=(
                len(config["vqvae_aug_conditioning_num_tokens"])
                + len(config["conditioning_num_tokens"])
            )
            if config["conditioning_type"]
            == TransformerConditioningType.PREPENDING.value
            else 0,
        )
    else:
        config["relative_spatial_pos_attr"] = None
        print("RelativeSpatialPositioning not yet implemented for 2D data")

    network = get_transformer_network(config).to(device)
    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=True,
            bucket_cap_mb=25,
            gradient_as_bucket_view=config["gradient_as_bucket_view"],
        )

    loss_function = CELoss().to(device)

    if config["device"] == "ddp" and config["use_zero"]:
        optimizer = ZeroRedundancyOptimizer(
            network.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config["learning_rate"],
            parameters_as_bucket_view=config["parameters_as_bucket_view"],
        )
    else:
        optimizer = torch.optim.Adam(network.parameters(), config["learning_rate"])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(
            config["gamma"]
            if isinstance(config["gamma"], float)
            else get_gamma(config=config, logger=logger)
        ),
    )

    train_handlers = [LoggingPreparationHandler()] if config["rank"] == 0 else []

    train_handlers += [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=False, epoch_level=False)
    ]

    if config["rank"] == 0:
        train_handlers += [
            LossSummaryHandler(loss=loss_function),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "train/"),
                interval=config["log_every"],
                epoch_level=True,
            ),
        ]

    key_metric = {
        f"Metric-CE-Prediction": CE(
            output_transform=lambda network_output: (
                network_output[CommonKeys.PRED],
                network_output[CommonKeys.LABEL],
            )
        )
    }

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=config["epochs"],
        non_blocking=True,
        train_data_loader=training_loader,
        network=network,
        optimizer=optimizer,
        inferer=TransformerTrainingInferer(),
        loss_function=loss_function,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch=batch,
            index_sequence=network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            vocab_size=config["vocab_size"],
            conditionings=config["conditionings"],
            use_continuous_conditioning=config["use_continuous_conditioning"],
            vqvae_network=config["vqvae_network"],
            vqvae_net_level=config["vqvae_net_level"],
            vqvae_aug_conditionings=config["vqvae_aug_conditionings"],
            device=pb_device,
            non_blocking=non_blocking,
        ),
        train_handlers=train_handlers,
        amp=False,
    )

    validation_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch=batch,
            index_sequence=network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            vocab_size=config["vocab_size"],
            conditionings=config["conditionings"],
            use_continuous_conditioning=config["use_continuous_conditioning"],
            vqvae_network=config["vqvae_network"],
            vqvae_net_level=config["vqvae_net_level"],
            vqvae_aug_conditionings=config["vqvae_aug_conditionings"],
            device=pb_device,
            non_blocking=non_blocking,
            trace_key_postfix="trace_dict",
        ),
        key_val_metric=key_metric,
        inferer=TransformerTrainingInferer(),
        amp=False,
        val_handlers=[
            LoggingPreparationHandler(),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "val_eval/"),
                interval=1,
                epoch_level=True,
                clamp_images=True,
                clamp_range=(0.0, 1.0),
                global_step_transform=lambda x: trainer.state.epoch,
            ),
        ]
        if config["rank"] == 0
        else [],
    )

    training_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=training_evaluation_loader,
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
            batch=batch,
            index_sequence=network.module.ordering.get_sequence_ordering()
            if config["device"] == "ddp"
            else network.ordering.get_sequence_ordering(),
            vocab_size=config["vocab_size"],
            conditionings=config["conditionings"],
            use_continuous_conditioning=config["use_continuous_conditioning"],
            vqvae_network=config["vqvae_network"],
            vqvae_net_level=config["vqvae_net_level"],
            vqvae_aug_conditionings=config["vqvae_aug_conditionings"],
            device=pb_device,
            non_blocking=non_blocking,
            trace_key_postfix="trace_dict",
        ),
        key_val_metric=key_metric,
        inferer=TransformerTrainingInferer(),
        amp=False,
        val_handlers=[
            LoggingPreparationHandler(),
            TensorBoardHandler(
                summary_writer=SummaryWriter(config["logs_directory"] + "train_eval/"),
                interval=1,
                epoch_level=True,
                clamp_images=True,
                clamp_range=(0.0, 1.0),
                global_step_transform=lambda x: trainer.state.epoch,
            ),
        ]
        if config["rank"] == 0
        else [],
    )

    to_save = {
        "network": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
    }

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        CheckpointLoader(
            load_path=config["checkpoint_fp"], load_dict=to_save, map_location=device
        ).attach(trainer)

    if config["device"] == "ddp" and config["use_zero"]:
        ConsolidateZeROHandler(
            zero_optimizer=optimizer,
            call_every=config["checkpoint_every"],
            recipient_rank=0,
            epoch_level=True,
        ).attach(trainer)

        ConsolidateZeROHandler(
            zero_optimizer=optimizer,
            call_every=config["checkpoint_every"],
            recipient_rank=0,
            epoch_level=True,
        ).attach(validation_evaluator)

    if config["rank"] == 0:
        CheckpointSaver(
            save_dir=config["checkpoint_directory"],
            save_dict=to_save,
            epoch_level=True,
            save_interval=config["checkpoint_every"],
            n_saved=2,
        ).attach(trainer)

        CheckpointSaver(
            save_dir=config["checkpoint_directory"],
            save_dict=to_save,
            epoch_level=True,
            save_key_metric=True,
            key_metric_name=validation_evaluator.state.key_metric_name,
            key_metric_n_saved=2,
        ).attach(validation_evaluator)

    MaxEpochsHandler(max_epochs=config["epochs"]).attach(trainer)

    if config["eval_every"] != 0:
        EvaluationHandler(
            evaluation_engine=validation_evaluator, evaluate_every=config["eval_every"]
        ).attach(trainer)

        EvaluationHandler(
            evaluation_engine=training_evaluator, evaluate_every=config["eval_every"]
        ).attach(trainer)

    if config["rank"] == 0:
        GpuInfo().attach(trainer, name="gpu")

        ProgressBar(
            persist=True,
            bar_format="[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{postfix} [{elapsed}<{remaining}]",
        ).attach(
            trainer,
            metric_names=["gpu:0 mem(%)", "gpu:0 util(%)"],
            output_transform=lambda output: {"Loss": output[CommonKeys.LOSS]},
        )

    trainer.run()

    torch.save(
        network.state_dict(),
        f"{config['checkpoint_directory']}model_state_dict_epoch={trainer.state.epoch}.pt",
    )


def inference(config: dict):
    logger, device = basic_initialization(
        config=config, logger_name="Transformer-Training"
    )

    if config["conditionings"]:
        config["use_continuous_conditioning"] = (
            list(config["use_continuous_conditioning"])
            if type(config["use_continuous_conditioning"]) is tuple
            else [config["use_continuous_conditioning"]] * len(config["conditionings"])
        )
    else:
        config["use_continuous_conditioning"] = []

    _, evaluation_loader, _ = get_data_flow(config=config, logger=logger)

    if config["vqvae_checkpoint"]:
        vqvae_network = get_vqvae_network(
            config={
                "network": config["vqvae_network"],
                "use_subpixel_conv": config["vqvae_net_use_subpixel_conv"],
                "use_slim_residual": config["vqvae_net_use_slim_residual"],
                "no_levels": config["vqvae_net_no_levels"],
                "downsample_parameters": config["vqvae_net_downsample_parameters"],
                "upsample_parameters": config["vqvae_net_upsample_parameters"],
                "no_res_layers": config["vqvae_net_no_res_layers"],
                "no_channels": config["vqvae_net_no_channels"],
                "codebook_type": config["vqvae_net_codebook_type"],
                "num_embeddings": config["vqvae_net_num_embeddings"],
                "embedding_dim": config["vqvae_net_embedding_dim"],
                "embedding_init": config["vqvae_net_embedding_init"],
                "commitment_cost": config["vqvae_net_commitment_cost"],
                "decay": config["vqvae_net_decay"],
                "norm": config["vqvae_net_norm"],
                "dropout": config["vqvae_net_dropout"],
                "act": config["vqvae_net_act"],
                "output_act": config["vqvae_net_output_act"],
                "data_spatial_dimension": config["vqvae_net_data_spatial_dimension"],
                "data_num_channels": config["vqvae_net_data_num_channels"],
            }
        )
        vqvae_network.to(device)
        Checkpoint.load_objects(
            to_load={"network": vqvae_network},
            checkpoint=torch.load(config["vqvae_checkpoint"], map_location=device),
            strict=True,
        )

    else:
        vqvae_network = None

    config["vqvae_network"] = vqvae_network

    temp_batch = next(iter(evaluation_loader))

    if config["vqvae_checkpoint"]:
        config["spatial_shape"] = (
            config["vqvae_network"]
            .index_quantize(temp_batch["quantization"].to(device))[
                config["vqvae_net_level"]
            ]
            .shape[1:]
        )

        if (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.BINARY.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = [2] * temp_batch[
                "quantization_binary_trace_dict"
            ]["total_count"].numpy().mean().astype(np.int)
        elif (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.CONTINUOUS.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = [-1] * temp_batch[
                "quantization_continuous_trace_dict"
            ]["total_count"].numpy().mean().astype(np.int)
        elif (
            config["vqvae_aug_conditionings"]
            == TransformerVQVAEConditioningTypes.NONE.value
        ):
            config["vqvae_aug_conditioning_num_tokens"] = []
        else:
            raise ValueError(
                f"VQVAE augmentation conditioning unknown. Received {config['vqvae_aug_conditionings']}, but valid "
                + f"options are {[e.value for e in TransformerVQVAEConditioningTypes]}"
            )
    else:
        config["spatial_shape"] = temp_batch["quantization"].shape[1:]
        config["vqvae_aug_conditioning_num_tokens"] = []

    config["ordering"] = Ordering(
        ordering_type=config["ordering_type"],
        spatial_dims=3,
        dimensions=(1,) + config["spatial_shape"],
        reflected_spatial_dims=config["reflected_spatial_dims"],
        transpositions_axes=config["transpositions_axes"],
        rot90_axes=config["rot90_axes"],
        transformation_order=config["transformation_order"],
    )

    config["relative_spatial_pos_attr"] = RelativeSpatialPositioning(
        spatial_dims=3,
        dimensions=(1,) + config["spatial_shape"],
        ordering=config["ordering"].get_sequence_ordering(),
        bucket_values=config["bucket_values"],
        bucket_beta=config["spatial_bias_max_dist"],
        conditioning_length=(
            len(config["vqvae_aug_conditioning_num_tokens"])
            + len(config["conditioning_num_tokens"])
        )
        if config["conditioning_type"] == TransformerConditioningType.PREPENDING.value
        else 0,
    ).to(device)

    network = get_transformer_network(config).to(device)
    log_network_size(network=network, logger=logger)

    if config["device"] == "ddp":
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=True,
            bucket_cap_mb=25,
            gradient_as_bucket_view=config["gradient_as_bucket_view"],
        )

    engine = SupervisedEvaluator(
        device=device,
        val_data_loader=evaluation_loader,
        inferer=TransformerInferenceInferer(
            sample=config["sample"],
            temperature=config["temperature"],
            top_k=config["top_k"],
        ),
        non_blocking=True,
        network=network,
        prepare_batch=lambda batch, pb_device, non_blocking: prepare_inference_batch(
            batch=batch,
            num_embeddings=config["vocab_size"],
            conditionings=config["conditionings"],
            use_continuous_conditioning=config["use_continuous_conditioning"],
            vqvae_network=config["vqvae_network"],
            vqvae_aug_conditionings=config["vqvae_aug_conditionings"],
            device=pb_device,
            non_blocking=non_blocking,
        ),
        amp=False,
        val_handlers=[],
    )

    if config.get("checkpoint_fp", None):
        # The warning is due to bad type hinting from MONAI. Internally the map_location is passed as follows
        #   checkpoint = torch.load(self.load_path, map_location=self.map_location)
        #   torch.load allows map_location to be a torch.device, so the following code is valid.
        CheckpointLoader(
            load_path=config["checkpoint_fp"],
            load_dict={"network": network},
            map_location=device,
        ).attach(engine)

    # TODO: Improve functionality by decorelating the inference from pre-existing encodings, needs data loading to use
    #   a CSV reader to load the conditionings and then use the subject column as file name.
    NpySaver(
        output_dir=config["outputs_directory"],
        output_postfix="sample",
        dtype=np.dtype(np.uint16),
        batch_transform=lambda batch: {
            "filename_or_obj": batch["quantization_meta_dict"]["filename_or_obj"]
            if config["additional_samples_multiplier"] == 0
            else [
                f.replace(".npy", f"_{i}.npy")
                for f, i in zip(
                    batch["quantization_meta_dict"]["filename_or_obj"],
                    batch["additional_sample_id"],
                )
            ]
        },
        output_transform=lambda output: output[CommonKeys.PRED],
    ).attach(engine)

    ProgressBar().attach(engine, output_transform=lambda output: {"Loss": 0})

    engine.run()


def run(
    # File system Parameters
    training_subjects: str,
    validation_subjects: str,
    project_directory: str,
    experiment_name: str,
    outputs_directory: str = None,
    mode: str = TransformerModes.TRAINING.value,
    additional_samples_multiplier: int = 0,
    conditioning_path: str = None,
    conditionings: Tuple[str, ...] = None,
    # Hardware Parameters
    device: int = 0,
    distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
    deterministic: bool = False,
    cuda_benchmark: bool = False,
    cuda_enabled: bool = True,
    use_zero: bool = False,
    gradient_as_bucket_view: bool = True,
    parameters_as_bucket_view: bool = True,
    seed: int = 2,
    # Training Parameters
    epochs: int = 1000000,
    learning_rate: float = 1e-4,
    gamma: Union[str, float] = "auto",
    log_every: int = 25,
    checkpoint_every: int = 50,
    eval_every: int = 50,
    weighted_sampling: bool = False,
    # VQVAE Augmentation Parameters
    vqvae_checkpoint: str = None,
    vqvae_aug_conditionings: str = TransformerVQVAEConditioningTypes.NONE.value,
    vqvae_aug_load_nii_canonical: bool = False,
    vqvae_aug_augmentation_probability: float = 0.2,
    vqvae_aug_augmentation_strength: float = 0.0,
    vqvae_aug_normalize: bool = True,
    vqvae_aug_standardize: bool = False,
    vqvae_aug_roi: Union[
        Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None,
    # VQVAE Network Parameters
    vqvae_network: str = "baseline_vqvae",
    vqvae_net_level: int = 0,
    vqvae_net_use_subpixel_conv: bool = False,
    vqvae_net_use_slim_residual: bool = True,
    vqvae_net_no_levels: int = 3,
    vqvae_net_downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = (
        (4, 2, 1, 1),
        (4, 2, 1, 1),
        (4, 2, 1, 1),
    ),
    vqvae_net_upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
        (4, 2, 1, 0, 1),
    ),
    vqvae_net_no_res_layers: int = 1,
    vqvae_net_no_channels: int = 128,
    vqvae_net_codebook_type: str = "ema",
    vqvae_net_num_embeddings: Tuple[int, ...] = (32,),
    vqvae_net_embedding_dim: Tuple[int, ...] = (64,),
    vqvae_net_embedding_init: Tuple[str, ...] = ("normal",),
    vqvae_net_commitment_cost: Tuple[float, ...] = (0.25,),
    vqvae_net_decay: Tuple[float, ...] = (0.99,),
    vqvae_net_norm: str = None,
    vqvae_net_dropout: float = 0.1,
    vqvae_net_act: str = "RELU",
    vqvae_net_output_act: str = None,
    vqvae_net_data_spatial_dimension: int = 3,
    vqvae_net_data_num_channels: int = 1,
    # Inference Parameters
    starting_epoch: int = 0,
    evaluation_checkpoint: str = "recent",
    sample: bool = True,
    temperature: float = 1.0,
    top_k: int = None,
    # Dataset Parameters
    batch_size: int = 2,
    eval_batch_size: int = 2,
    num_workers: int = 8,
    prefetch_factor: int = 6,
    # Sequence Ordering Parameters
    ordering_type: str = OrderingType.RASTER_SCAN.value,
    reflected_spatial_dims: Union[Tuple[bool, bool], Tuple[bool, bool, bool]] = (
        False,
        False,
        False,
    ),
    transpositions_axes: Union[
        Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
    ] = tuple(),
    rot90_axes: Union[
        Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
    ] = tuple(),
    transformation_order: Tuple[
        OrderingTransformations, OrderingTransformations, OrderingTransformations
    ] = (
        OrderingTransformations.TRANSPOSE.value,
        OrderingTransformations.ROTATE_90.value,
        OrderingTransformations.REFLECT.value,
    ),
    # General Transformers Parameters
    network: str = "performer",
    vocab_size: int = 32,
    n_embd: int = 256,
    n_layers: int = 10,
    n_head: int = 8,
    tie_embedding: bool = False,
    ff_glu: bool = False,
    use_scalenorm: bool = False,
    emb_dropout: float = 0.0,
    ff_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    use_rezero: bool = False,
    position_emb: str = "absolute",
    spatial_position_emb: str = None,
    conditioning_type: str = TransformerConditioningType.NONE.value,
    use_continuous_conditioning: Union[bool, Tuple[bool, ...]] = False,
    # Performer Parameters
    local_attn_heads: int = 0,
    local_window_size: int = 256,
    feature_redraw_interval: int = 1000,
    generalized_attention: bool = False,
    # XTransformer Parameters
    use_rmsnorm: bool = False,
    attn_talking_heads: bool = False,
    attn_gate_values: bool = False,
    attn_on_attn: bool = False,
    sandwich_coef: int = None,
    sandwich_norm: bool = False,
    rel_pos_bias: bool = False,
    spatial_rel_pos_bias: bool = False,
    bucket_values: bool = False,
    spatial_bias_max_dist: int = 50,
    use_qk_norm_attn: bool = False,
    residual_attn: bool = False,
    cross_residual_attn: bool = False,
    gate_residual: bool = False,
    shift_mem_down: int = 0,
    data_spatial_dimension: int = 3,
):
    f"""
    Entry point for the transformer handling. It follows this structure since it is the same one found in the
    Distributed Data Parallelism Ignite tutorial found at :

    https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10

    Args:
        -------------------------------------------- File system Parameters --------------------------------------------

        training_subjects (str): Path towards either a folder with .npy files or towards a csv/tsv which has a 'path'
            column that stores full paths towards .npy files. Those will be used for training.

        validation_subjects (str): Path towards either a folder with .npy files or towards a csv/tsv which has a 'path'
            column that stores full paths towards .npy files. Those will be used for validation.

        conditioning_path (str): Path towards a csv/tsv file that has a 'subject' column in which the file names from
            both training and validation subjects are and the other columns hold conditioning information

        conditionings (Tuple[str,...]): The conditionings from the conditioning_path files that will be prepended to the
            transformer input. The elements of the Tuple must be column names from the file.

        project_directory (str): Path towards folder where the experiment folder will be created.

        experiment_name (str): Name of the experiment which will be used to name a folder in the project_directory.
            Defaults to 'nvidia'.

        outputs_directory (str): Force overwriting the default outputs directory. Defaults to None.

        mode (str) : It can be one of the following: {[e.value for e in TransformerModes]}.
            'training': Given the location of the .npy quantization representations the configured transformer will be
                trained.
            'inference': In this mode random samples will be generated. The number of samples will be equal to the size
                of the validation dataset.

        additional_samples_multiplier (int): For how many additional times to sample based on the same sample's
            conditioning. Defaults to 0.

        --------------------------------------------- Hardware parameters ----------------------------------------------

        device (int): The index of the GPU in the PCI_BUS_ID order. Defaults to 0.

        distributed_port (int): Torch distributed backend port. Defaults to 29500.

        deterministic (bool): Boolean that sets monai.utils.set_determinism. This should be set to False for 
            transformers to load their states correctly. Defaults to False.

        cuda_benchmark (bool): Boolean that sets whether cuda_benchmark will be used. It is not exclusive with
            deterministic, but it supersedes it. This should be set to False for transformers to load their states 
            correctly. Defaults to False.
            
        cuda_enabled (bool): A bool that controls whether cuDNN is enabled. This should be set to True for transformers
            to load their states correctly. Defaults to True.
            
        use_zero (bool): Whether or not to use Zero Redundancy Optimizer. Defaults to False.

        gradient_as_bucket_view (bool): When set to True, gradients will be views pointing to different offsets of
            allreduce communication buckets. This can reduce peak memory usage, where the saved memory size will be
            equal to the total gradients size. Defaults to True.

        parameters_as_bucket_view (bool): If True, parameters are packed into buckets to speed up communication, and 
            param.data fields point to bucket views at different offsets. Defaults to True.

        seed (int): The seed to be used for the experiment. Defaults to 2.

        --------------------------------------------- Training Parameters ----------------------------------------------

        epochs (int): Number of epochs that the network will be trained. Defaults to 1,000,000.

        learning_rate (float): Learning rate of the optimizer. Defaults to 1e-4.

        gamma (Union[str,float]): Gamma that will be used for learning rate decay. Defaults to 'auto' which calculates
            the decay so that at the end of the training the learning rate is equal to 1e-5.

        log_every (int): After how many epochs we save the logs. Defaults to 25.

        checkpoint_every (int): After how many epochs we save a checkpoint. Defaults to 50.

        eval_every (int): After how many epochs do we run evaluation. Defaults to 50.
        
        weighted_sampling (bool): Whether or not to use weighted sampling for training. It needs a conditioning_path to 
            be passed in which a subject column with the file name and a weight column with the subject's weight can be 
            found. Defaults to False.

        ---------------------------------------- VQVAE Augmentation Parameters -----------------------------------------

        vqvae_checkpoint (str): Path to the vqvae checkpoint to be loaded and used to generated the augmented
            representations. Defaults to None.

        vqvae_aug_conditionings (str): Whether or not the transformer is conditioned on the VQ-VAE augmentations. The
            possible values are:
                "none" for no VQ-VAE augmentation
                "binary" for a binary conditioning of which augmentations have been applied to each sample
                "continuous" for a continuous conditioning based on internal parameters (where possible otherwise binary)
                    of each augmentation that has been applied to each sample
            Defaults to "none".

        vqvae_aug_load_nii_canonical (bool): If true will reorder image array data when loading .nii images to be as
            closest to canonical. Defaults to True.

        vqvae_aug_augmentation_probability (float): The probabilities of every augmentation. Defaults to 0.2

        vqvae_aug_augmentation_strength (float): The multiplier of the ADDED augmentation strength. It defaults to 0 for
            no added strength. Augmentations' strength increments are defined in utils.vqvae.AugmentationStrengthScalers.
            Defaults to 0.

        vqvae_aug_normalize (bool): Whether to normalize the input data in the 0-1 range. normalize and standardize
            can't be turned on simultaneously. Defaults to True.

        vqvae_aug_standardize (bool): Whether to standardize the input data. normalize and standardize can't be turned on
            simultaneously. Defaults to False.

        vqvae_aug_roi (Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]): What is
            the region of interest in the image. Will be given to monai.transforms.CenterSpatialCropd if tree ints are
            passed, otherwise it will be given to monai.transforms.SpatialCropd if three tuples (start, end) are given.
            Defaults to None.

        ------------------------------------------- VQVAE Network Parameters -------------------------------------------

        vqvae_network (str): What vqvae network to use. Defaults to 'baseline_vqvae'.

        vqvae_net_level (int): Which level of the VQ-VAE the transformer will be trained on. Defaults to 0.

        vqvae_net_use_subpixel_conv (bool): Whether or not to use SubPixelConvolution as the last transpose convolution
            in the network. Defaults to True.

        vqvae_net_use_slim_residual (bool): Whether or not to have the kernel of the last convolution in each residual
            unit be equal to 1. Default to True

        vqvae_net_no_levels (int): How many levels the VQVAE has. Defaults to 3.

        vqvae_net_downsample_parameters (Tuple[Tuple[int,int,int,int],...]): A Tuple of Tuples for defining the
            downsampling convolutions. Each Tuple should hold the following information kernel_size (int), stride (int),
            padding (int), dilation(int). Defaults to ((4,2,1,1),(4,2,1,1),(4,2,1,1)).

        vqvae_net_upsample_parameters (Tuple[Tuple[int,int,int,int,int],...]): A Tuple of Tuples for defining the
            upsampling convolutions. Each Tuple should hold the following information kernel_size (int), stride (int),
            padding (int), output_padding (int), dilation(int). Defaults to ((4,2,1,0,1),(4,2,1,0,1),(4,2,1,0,1)).

        vqvae_net_no_res_layers (int): How many residual layers we use per level. Defaults to 1.

        vqvae_net_no_channels (int): How many channels the deepest level has. Defaults to 128.

        vqvae_net_codebook_type (ema): What codebook type to use. Defaults to 'ema'.

        vqvae_net_num_embeddings (Tuple[int,...]): How many codebook elements (atomic elements/tokens) each of the
            embedding spaces has. Defaults to (32, ).

        vqvae_net_embedding_dim (Tuple[int,...]): The channel dimension size of the elements of each codebook.
            Defaults to (64, ).

        vqvae_net_embedding_init (Tuple[str,...]): The initialization used for the codebook elements. Can be either
            'normal' or 'kaiming_uniform'. Defaults to ('normal',).

        vqvae_net_commitment_cost (Tuple[float,...]): The commitment cost factor that will be used to scale the EMA
            latent loss. Defaults to (0.25, ).

        vqvae_net_decay (Tuple[float, ...]): The decay factor that will be used to update the embedding space when EMA
            Quantization update is being used. Defaults to (0.99, ).

        vqvae_net_norm (str): Which normalization technique the network will use. Defaults to None.

        vqvae_net_dropout (float): The amount of dropout use in the network. Defaults to 0.1

        vqvae_net_act (str): Which activation function the network uses. Defaults to 'RELU'.

        vqvae_net_output_act (str): Which activation function should the output be passed through. Defaults to None.
        
        vqvae_net_data_spatial_dimension (int): Is your input data 2D or 3D. Defaults to 3.

        vqvae_net_data_num_channels (int): how many input/output channels for your data. Defaults to 1.

        --------------------------------------------- Inference Parameters ---------------------------------------------

        starting_epoch (int): At which epoch we start the training. Defaults to 0.

        evaluation_checkpoint (str): Choose which checkpoint to use when performing inference. "recent" uses the most
            recent available, and "best" uses the checkpoint that achieved the best performance according to the key metric.
            Defaults to "recent".

        sample (bool): Whether the values are sampled from the distribution or take the most likely. Defaults to True.

        temperature (float): Temperature value to scale the logits by 1/temperature. Defaults to 1.0.

        top_k (int): Crop probabilities to only the top k options. If None, no cropping happens. Defaults to None.

        ---------------------------------------------- Dataset Parameters ----------------------------------------------

        batch_size (int): The batch size that will be used during training. Defaults to 2.

        eval_batch_size (int): The batch size that will be used during evaluation. Defaults to 2.

        num_workers (int): The number of threads that will be used to load batches. Defaults to 8.

        prefetch_factor (int): How may batches each thread will try and buffer. Defaults to 6.
        
        data_spatial_dimension (int): Is your input data 2D or 3D. Defaults to 3.

        ----------------------------------------- Sequence Ordering Parameters -----------------------------------------

        ordering_type (str): The ordering logic that will be applied to project from 2D/3D tensor to 1D tensor. It can
            be one of the following: {[e.value for e in OrderingType]}. Defaults to 'raster_scan'.

        reflected_spatial_dims (Union[Tuple[bool, bool], Tuple[bool, bool, bool]]): Weather or not to flip axes of the
            2D/3D tensor before being projected to a 1D tensor. Defaults to (False, False, False).

        transpositions_axes (Union[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]]): Around which axes to
            apply np.transpose. Defaults to ().

        rot90_axes (Union[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]]): Around which axes to apply
            np.rot90. Defaults to ().

        transformation_order (Tuple[OrderingTransformations, OrderingTransformations, OrderingTransformations]): In
            which order the np.transpose, np.rot90, np.reflect are being called. It should contain at least one and at
            most all the following: {[e.value for e in OrderingTransformations]}. Defaults to
            ("transpose", "rotate_90", "reflect").

        ---------------------------------------- General Transformers Parameters ---------------------------------------

        network (str): What network to use. Defaults to 'performer'.

        vocab_size (int): The size of the vocabulary. It must be the same values as the "num_embeddings" argument used
            during the vqvae training. Defaults to 32.

        n_embd (int): The size of the latent representation that the transformer will use. Defaults to 256.

        n_layers (int): The number of layers the transformer will have. Defaults to 10.

        n_head (int): The number of heads that the self attention mechanism will use.

        tie_embedding (bool): Whether or not to have a Linear layer to project the transformer outputs into logits.
            Defaults to False.

        ff_glu (bool): Gating with GELU the feedforward layers. Defaults to False.

        use_scalenorm (bool): Replacing the LayerNorm with a scaled L2 regularisation. Defaults to False.

        emb_dropout (float): Drop probability for the Dropout layer just after the embedding layer.

        ff_dropout (float): Drop probability for the Dropout layer just after the linear layers.

        attn_dropout (float): Drop probability for the Dropout layer just after the attention mechanism.

        use_rezero (bool): Whether or not to use Rezero logic for improved convergence. Defaults to False.

        position_emb (str): It can be either None for no spatial positioning or 'fixed', 'absolute' or 'rotary'.

        spatial_position_emb (str): It can be either None for no spatial positioning or 'fixed' or 'absolute'.
            Defaults to None.

        conditioning_type (str): The style of conditioning that will be used in the transformer. It can be one of the
            following:
            'none' for no conditioning
            'bos_replacement' where the beginning of sentence token is replaced by the summation of the conditionings
            'prepending' where the conditionings' embeddings are prepended before the beginning of sentence token
            'cross_attend' where cross attention is used to inject the conditioning at every level of the model

        use_continuous_conditioning (Tuple[bool,...]): Whether or not to tile the values under the assumption they are
            continuous or pass the conditioning values through a nn.Embedding assuming the values are already quantised.
            The VQ-VAE augmentations conditioning are treated as quantised conditionings. Defaults to False.

        --------------------------------------------- Performer Parameters ---------------------------------------------

        local_attn_heads (int): How many of the n_head will be local attention heads instead of global attention ones.
            Defaults to 0.

        local_window_size (int): The number of tokens the local attention heads will look at. Defaults to 256.

        feature_redraw_interval (int): How frequently to redraw the projection matrix, the more frequent, the slower
            the training. Defaults to 1000.

        generalized_attention (bool): Whether or not to use generalized attention or the softmax approximation.
            Defaults to False.

        -------------------------------------------- XTransformer Parameters -------------------------------------------

        use_rmsnorm (bool): Replace layer normalization with a simpler alternative, without mean centering and the
            learned bias. Defaults to False.

        attn_talking_heads (bool): Mixing information between heads pre and post attention (softmax). Defaults to False.

        attn_on_attn (bool): Adds a gated linear unit at the end of the attention layer, further gated by the original
            queries. Defaults to False.

        attn_gate_values (bool): Peculiar variant of attention where the aggregated values are gated with the input.
            Defaults to False.

        sandwich_coef (int):  How many blocks of only attention at the beginning followed by blocks of feedforwards at
            the end. Nvidia found 6 to be the optimal values. Defaults to None.

        sandwich_norm (bool): When using pre-layernorm, to add an extra layernorm to all the branch outputs. Effective
            when facing instability during training. Defaults to False.

        rel_pos_bias (bool): Relative positional encoding based on learned bias values that are added to the attention
            matrix pre-softmax. Defaults to False.

        spatial_rel_pos_bias (bool): Spatial Relative positional bias using spatial information from the original
            3D latent space based on a learned bias added to the attention matrix pre-softmax. Defaults to False

        bucket_values (bool): Whether to bucket distance values in spatial relative positional bias by a fixed
            distance. i.e. codes beyond a certain distance will have the same bias term. Defaults to False

        spatial_bias_max_dist (int): only relavent if bucket_values is True. Determines the maximum distance
            at which point distance bias is the same for spatial relative positional bias. Default to 50.

        use_qk_norm_attn (bool): L2 normalize the queries and keys along the head dimension before the dot product.
            Defaults to False.

        residual_attn (bool):  Whether or not to use Attention Score residual connection between attention modules.
            Defaults to False.

        cross_residual_attn (bool): Whether or not to use Attention Score residual connection between cross-attention
            modules. Defaults to False.

        gate_residual (bool): Whether or not to gate the residual connections. Defaults to False.

        shift_mem_down (int): They simply route the memory segment of a layer to the layer below it, for the next
            recurrent step. Defaults to 0.
            


    """
    config = locals()

    modes = [m.value for m in TransformerModes]

    if config["device"] == "ddp":
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=True,
            verbose=False,
            init_method=None,
            distributed_port=config["distributed_port"],
        )

        config["rank"] = int(os.environ["RANK"])
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        config["rank"] = 0
        config["local_rank"] = 0
        config["world_size"] = 1

    if config["mode"] == TransformerModes.TRAINING.value:
        training(config=config)
    elif config["mode"] == TransformerModes.INFERENCE.value:
        inference(config=config)
    else:
        raise ValueError(
            f"Transformer mode unknown. Was given {config['mode']} but choices are {modes}."
        )


if __name__ == "__main__":
    Fire({"run": run})
