import os
from copy import deepcopy
from logging import Logger
from typing import Dict, Tuple, Sequence, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from ignite.utils import convert_tensor
from monai.data import Dataset, DataLoader, DistributedSampler,DistributedWeightedRandomSampler
from monai.transforms import ToTensord, Compose
from monai.transforms.io.dictionary import LoadImaged
from torch import nn

from src.networks.vqvae.vqvae import VQVAEBase
from src.utils.constants import (
    VQVAEModes,
    TransformerModes,
    TransformerVQVAEConditioningTypes,
)
from src.utils.vqvae import get_transformations as get_vqvae_transformations


class AbsoluteSpatialPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, spatial_indices_sequence: torch.Tensor):
        super().__init__()

        self.register_buffer("spatial_indices_sequence", spatial_indices_sequence)
        # Eliminating the last element as it is the predicted one
        self.spatial_indices_sequence = self.spatial_indices_sequence[:-1]

        self.padding = lambda x: F.pad(x, (0, 0, 1, 0, 0, 0), "constant", 0)

        self.emb = nn.Embedding(len(self.spatial_indices_sequence), dim)

    def forward(self, x):
        sc = self.emb(self.spatial_indices_sequence)
        sc = sc[None, : x.shape[1] - 1, :].to(x)
        sc = self.padding(sc)

        return sc


class FixedSpatialPositionalEmbedding(nn.Module):
    def __init__(self, dim, spatial_indices_sequence):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        max_position = torch.max(spatial_indices_sequence)
        # The + 1 is for torch.arange to include the max_position as well
        position = torch.arange(0, max_position + 1, dtype=torch.float)

        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        sinusoid_inp = sinusoid_inp[spatial_indices_sequence, :]

        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # Eliminating the last element as it is the predicted one
        emb = emb[:-1]
        self.register_buffer("emb", emb)

        self.padding = lambda x: F.pad(x, (0, 0, 1, 0, 0, 0), "constant", 0)

    def forward(self, x):
        sc = self.emb
        # The sc is cropped for just the amount of spatial information
        sc = sc[None, : x.shape[1] - 1, :].to(x)
        sc = self.padding(sc)
        return sc


def get_transformations(
    use_vqvae_augmentations: bool, vqvae_transformations_config: dict
):
    vqvae_augmentations = None
    if use_vqvae_augmentations:
        transform, vqvae_augmentations = get_vqvae_transformations(
            **vqvae_transformations_config
        )
    else:
        transform = [
            LoadImaged(keys=["quantization"]),
            ToTensord(keys=["quantization"]),
        ]

    return Compose(transform), vqvae_augmentations


def get_subjects(
    subjects_file_path: str,
    conditioning_path: str,
    conditionings: Tuple[str],
    use_continuous_conditioning: Tuple[bool],
    additional_samples_multiplier: int,
    weighted_sampling: bool,
    logger: Logger,
) -> Tuple[List[Dict[str, str]], Dict[str, int], torch.Tensor]:
    if os.path.isdir(subjects_file_path):
        subjects_files = [
            os.path.join(subjects_file_path, os.fsdecode(f))
            for f in os.listdir(subjects_file_path)
        ]
    elif os.path.isfile(subjects_file_path):
        if subjects_file_path.endswith(".csv"):
            subjects_files = pd.read_csv(
                filepath_or_buffer=subjects_file_path, sep=","
            )["path"].to_list()
        elif subjects_file_path.endswith(".tsv"):
            subjects_files = pd.read_csv(
                filepath_or_buffer=subjects_file_path, sep="\t"
            )["path"].to_list()
    else:
        raise ValueError(
            "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
        )

    offsets = None
    if conditioning_path:
        if os.path.isfile(conditioning_path):
            if conditioning_path.endswith(".csv"):
                conditioning_file = pd.read_csv(
                    filepath_or_buffer=conditioning_path, sep=","
                )
            elif conditioning_path.endswith(".tsv"):
                conditioning_file = pd.read_csv(
                    filepath_or_buffer=conditioning_path, sep="\t"
                )
        else:
            raise ValueError("Path is not a csv/tsv with file paths inside.")

        offsets = {}
        for conditioning, is_continous in zip(
            conditionings, use_continuous_conditioning
        ):
            offsets[conditioning] = (
                -1 if is_continous else conditioning_file[conditioning].nunique()
            )

    subjects = []
    weights = []
    nan_subjects = 0
    mia_subjects = 0

    for file in subjects_files:
        valid_subject = True
        subject_name = os.path.basename(file)
        subject = {"quantization": file}
        if conditioning_path:
            for conditioning in conditionings:
                try:
                    conditioning_value = conditioning_file.loc[
                        conditioning_file["subject"] == subject_name, conditioning
                    ].values[0]
                except IndexError:
                    mia_subjects += 1
                    valid_subject = False
                    break

                if np.isnan(conditioning_value):
                    nan_subjects += 1
                    valid_subject = False
                    break

                subject[conditioning] = conditioning_value

            if weighted_sampling:
                weights.append(
                    conditioning_file.loc[
                        conditioning_file["subject"] == subject_name, "weight"
                    ].values[0]
                )

        if valid_subject:
            if additional_samples_multiplier != 0:
                for additional_sample_id in range(additional_samples_multiplier):
                    additional_subject = deepcopy(subject)
                    additional_subject["additional_sample_id"] = additional_sample_id
                    subjects.append(additional_subject)
            else:
                subjects.append(subject)

    if mia_subjects > 0 or nan_subjects > 0:
        logger.warning(
            f"{mia_subjects + nan_subjects} were discarded during data loading. "
            f"{mia_subjects} did not have matching conditioning and {nan_subjects} had conditioning that was NaN. "
            f"Make sure your conditioning data covers all of your subjects."
        )
    return subjects, offsets, torch.as_tensor(weights).double()


def get_data_flow(
    config: dict, logger: Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # TODO: Add the new parameters for the vqvae transformations
    """
    Constructs the data ingestion logic. The quantization element will be loaded at the key "quantization".

    The following fields are needed in config:
        training_subjects (str): Absolute path to either a folder, csv or tsv. If it is a folder all .nii.gz files
        will be ingested. If it is a csv or tsv, it is expected that a "path" column is present and holds
        absolute paths to individual .nii.gz files. Those subjects will be used for the training dataset.

        validation_subjects (str): Absolute path to either a folder, csv or tsv. If it is a folder all .nii.gz files
        will be ingested. If it is a csv or tsv, it is expected that a "path" column is present and holds
        absolute paths to individual .nii.gz files. Those subjects will be used for the validation dataset.

        batch_size (int): The batch size that will be used to train the network. Defaults to 2.

        eval_batch_size (int): The batch size that will be used to evaluate the network. Defaults to 2.

        num_workers (int): How many threads will be spawn to load batches. Defaults to 8.

        prefetch_factor (int): How many batches each thread will try to keep as a buffer. Defaults to 6.

        conditioning_path (str): Path towards a csv/tsv file that has a 'subject' column in which the file names
        from both training and validation subjects are and the other columns hold conditioning information

        conditionings (Tuple[str,...]): The conditionings from the conditioning_path files that will be prepended to
        the transformer input. The elements of the Tuple must be column names from the file.

        vocab_size (int): The size of the vocabulary. It must be the same values as the "num_embeddings" argument
        used during the vqvae training.

    Args:
        config (dict): Configuration dictionary that holds all the required parameters.

        logger (Logger): Logger that will be used to report DataLoaders parameters.

    Returns:
        DataLoader: Training DataLoader for the training data

        DataLoader: Evaluation DataLoader for the validation data.
    """

    training_transform, vqvae_augmentations = get_transformations(
        use_vqvae_augmentations=config["vqvae_checkpoint"] is not None,
        vqvae_transformations_config={
            "mode": VQVAEModes.TRAINING.value,
            "load_nii_canonical": config["vqvae_aug_load_nii_canonical"],
            "augmentation_probability": config["vqvae_aug_augmentation_probability"],
            "augmentation_strength": config["vqvae_aug_augmentation_strength"],
            "no_augmented_extractions": config["additional_samples_multiplier"],
            "num_embeddings": config["vqvae_net_num_embeddings"],
            "normalize": config["vqvae_aug_normalize"],
            "standardize": config["vqvae_aug_standardize"],
            "roi": config["vqvae_aug_roi"],
            "patch_size": None,
            "num_samples": 1,
            "key": "quantization",
            "data_num_channels": config["vqvae_net_data_num_channels"]
        },
    )

    training_subjects, offsets, weights = get_subjects(
        subjects_file_path=config["training_subjects"],
        conditioning_path=config["conditioning_path"],
        conditionings=config["conditionings"],
        use_continuous_conditioning=config["use_continuous_conditioning"],
        additional_samples_multiplier=config["additional_samples_multiplier"],
        weighted_sampling=config["weighted_sampling"],
        logger=logger,
    )

    training_dataset = Dataset(data=training_subjects, transform=training_transform)

    if config["device"] == "ddp" and config["weighted_sampling"]:
        sampler = DistributedWeightedRandomSampler(
            dataset=training_dataset,
            weights=weights,
            num_samples_per_rank=None,
            even_divisible=True,
            shuffle=True,
            drop_last=True
        )
    elif config["device"] == "ddp" and not config["weighted_sampling"]:
        sampler = DistributedSampler(
            dataset=training_dataset, shuffle=True, even_divisible=True,
        )
    elif config["device"] != "ddp" and config["weighted_sampling"]:
        sampler = WeightedRandomSampler(
            weights=weights,num_samples=len(training_dataset),replacement=False,generator=training_dataset
        )
    elif config["device"] != "ddp" and not config["weighted_sampling"]:
        sampler = None

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 8),
        shuffle=sampler is None,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        # Forcefully setting it to false due to this pull request
        # not being in PyTorch 1.7.1
        # https://github.com/pytorch/pytorch/pull/48543
        persistent_workers=False,
        sampler=sampler,
    )

    evaluation_subjects, _, _ = get_subjects(
        subjects_file_path=config["validation_subjects"],
        conditioning_path=config["conditioning_path"],
        conditionings=config["conditionings"],
        use_continuous_conditioning=config["use_continuous_conditioning"],
        additional_samples_multiplier=config["additional_samples_multiplier"],
        weighted_sampling=config["weighted_sampling"],
        logger=logger,
    )

    evaluation_transform, _ = get_transformations(
        use_vqvae_augmentations=config["vqvae_checkpoint"] is not None,
        vqvae_transformations_config={
            "mode": VQVAEModes.EXTRACTING.value,
            "load_nii_canonical": config["vqvae_aug_load_nii_canonical"],
            "augmentation_probability": config["vqvae_aug_augmentation_probability"],
            "augmentation_strength": config["vqvae_aug_augmentation_strength"],
            "no_augmented_extractions": 0,
            "num_embeddings": config["vqvae_net_num_embeddings"],
            "normalize": config["vqvae_aug_normalize"],
            "standardize": config["vqvae_aug_standardize"],
            "roi": config["vqvae_aug_roi"],
            "patch_size": None,
            "num_samples": 1,
            "key": "quantization",
            "data_num_channels": config["vqvae_net_data_num_channels"]
        },
    )

    evaluation_dataset = Dataset(
        data=evaluation_subjects, transform=evaluation_transform
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.get("eval_batch_size", 2),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=evaluation_dataset,
            shuffle=False,
            even_divisible=config["mode"] == TransformerModes.TRAINING,
        )
        if config["device"] == "ddp"
        else None,
    )

    training_evaluation_dataset = Dataset(
        data=training_subjects, transform=evaluation_transform
    )

    training_evaluation_loader = DataLoader(
        training_evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        sampler=DistributedSampler(
            dataset=evaluation_dataset, shuffle=False, even_divisible=False
        )
        if config["device"] == "ddp"
        else None,
    )

    if logger:
        logger.info("Dataflow setting:")
        logger.info("\tTraining:")
        logger.info(f"\t\tLength: {len(training_loader)}")
        logger.info(f"\t\tBatch Size: {training_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {training_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {training_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {training_loader.prefetch_factor}")
        logger.info("\tValidation:")
        logger.info(f"\t\tLength: {len(evaluation_loader)}")
        logger.info(f"\t\tBatch Size: {evaluation_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {evaluation_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {evaluation_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {evaluation_loader.prefetch_factor}")

    config["epoch_length"] = len(training_loader)

    conditioning_num_tokens = []
    if offsets:
        for c, o in offsets.items():
            conditioning_num_tokens.append(o)

    config["conditioning_num_tokens"] = conditioning_num_tokens

    config["vqvae_augmentations"] = vqvae_augmentations

    return training_loader, evaluation_loader, training_evaluation_loader


@torch.no_grad()
def prepare_batch(
    batch: Dict,
    index_sequence: Sequence[int],
    vocab_size: int,
    conditionings: Sequence[str] = None,
    use_continuous_conditioning: Sequence[bool] = None,
    vqvae_network: VQVAEBase = None,
    vqvae_net_level: int = 0,
    vqvae_aug_conditionings: str = TransformerVQVAEConditioningTypes.NONE.value,
    trace_key_postfix: str = "trace_dict",
    device: torch.device = None,
    non_blocking: bool = False,
):
    """
    Batch preparation logic of the quantization elements for training.

    If a VQ-VAE is pass then firstly the images are beign quantized than rasterized it through the reshape method
    followed by a reordering based on the index_sequence which is can be arbitrarily generated.

    After rasterizing a padding to the left with the vocab_size int is done since the quantization elements are
    actually in [0, vocab_size) natural numbers range.

    Then the processed encoding is split into input, which is everything but the last element and target which is
    everything but the first element which is added by padding.
    """
    encoded = batch["quantization"]
    encoded = convert_tensor(encoded, device, non_blocking)

    if vqvae_network:
        encoded = vqvae_network.index_quantize(encoded)[vqvae_net_level]

    encoded = encoded.reshape(encoded.shape[0], -1)
    encoded = encoded[:, index_sequence]
    encoded = F.pad(encoded, (1, 0), "constant", vocab_size)
    encoded = encoded.long()

    conditioned = None
    if (
        vqvae_network
        and vqvae_aug_conditionings != TransformerVQVAEConditioningTypes.NONE.value
    ):
        conditioned = []

        if vqvae_aug_conditionings == TransformerVQVAEConditioningTypes.BINARY.value:
            trace = batch[f"quantization_binary_{trace_key_postfix}"]["trace"]
        elif (
            vqvae_aug_conditionings
            == TransformerVQVAEConditioningTypes.CONTINUOUS.value
        ):
            trace = batch[f"quantization_continuous_{trace_key_postfix}"]["trace"]

        for transform_name, value in trace.items():
            if len(value.shape) == 1:
                value = value[..., None]

            if (
                vqvae_aug_conditionings
                == TransformerVQVAEConditioningTypes.BINARY.value
            ):
                value = value.long()
            elif (
                vqvae_aug_conditionings
                == TransformerVQVAEConditioningTypes.CONTINUOUS.value
            ):
                value = value.float()

            value = convert_tensor(value, device, non_blocking)
            conditioned.append(value)

    if conditionings:
        conditioned = [] if conditioned is None else conditioned

        for conditioning_label, continuous_conditioning in zip(
            conditionings, use_continuous_conditioning
        ):
            value = batch[conditioning_label]

            if len(value.shape) == 1:
                value = value[..., None]

            if not continuous_conditioning:
                value = value.long()
            else:
                value = value.float()

            value = convert_tensor(value, device, non_blocking)
            conditioned.append(value)

    x_input = convert_tensor(encoded[:, :-1], device, non_blocking)
    x_target = convert_tensor(encoded[:, 1:], device, non_blocking)

    return (x_input, conditioned), x_target


@torch.no_grad()
def prepare_inference_batch(
    batch: Dict,
    num_embeddings: int,
    conditionings: Sequence[str] = None,
    use_continuous_conditioning: Sequence[bool] = None,
    vqvae_network: VQVAEBase = None,
    vqvae_aug_conditionings: str = TransformerVQVAEConditioningTypes.NONE.value,
    trace_key_postfix: str = "trace_dict",
    device: torch.device = None,
    non_blocking: bool = False,
):
    """
    Batch preparation logic of the quantization elements for inference.

    Given loaded quantization the batch size is determined and no_samples of single value tensor are being generated
    where the value is the num_embedding since this was used as start of sentence token during the training.

    If a VQ-VAE network is being given, False (0) conditionings are being set for each of VQ-VAEs augmentation
    conditionings. Afterwards, the arbitrary conditionings are being added.
    """
    no_samples = batch["quantization"].shape[0]
    start_pixel = np.array([[num_embeddings]])
    start_pixel = np.repeat(start_pixel, no_samples, axis=0)
    initial = torch.from_numpy(start_pixel)
    initial = initial.long()

    conditioned = None
    if (
        vqvae_network
        and vqvae_aug_conditionings != TransformerVQVAEConditioningTypes.NONE.value
    ):
        conditioned = []

        if vqvae_aug_conditionings == TransformerVQVAEConditioningTypes.BINARY.value:
            trace = batch[f"quantization_binary_{trace_key_postfix}"]["trace"]
        elif (
            vqvae_aug_conditionings
            == TransformerVQVAEConditioningTypes.CONTINUOUS.value
        ):
            trace = batch[f"quantization_continuous_{trace_key_postfix}"]["trace"]

        for transform_name, value in trace.items():
            if len(value.shape) == 1:
                value = value[..., None]

            if (
                vqvae_aug_conditionings
                == TransformerVQVAEConditioningTypes.BINARY.value
            ):
                value = value.long()
            elif (
                vqvae_aug_conditionings
                == TransformerVQVAEConditioningTypes.CONTINUOUS.value
            ):
                value = value.float()

            value = convert_tensor(value, device, non_blocking)
            conditioned.append(value)

    if conditionings:
        conditioned = [] if conditioned is None else conditioned

        for conditioning_label, continuous_conditioning in zip(
            conditionings, use_continuous_conditioning
        ):
            value = batch[conditioning_label]

            if len(value.shape) == 1:
                value = value[..., None]

            if not continuous_conditioning:
                value = value.long()
            else:
                value = value.float()

            value = convert_tensor(value, device, non_blocking)
            conditioned.append(value)

    x_input = convert_tensor(initial, device, non_blocking)
    x_target = convert_tensor(initial, device, non_blocking)

    return (x_input, conditioned), x_target
