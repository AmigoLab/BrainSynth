from enum import Enum
from logging import Logger
from typing import List

import numpy as np

from src.handlers.general import ParamSchedulerHandler
from src.networks.vqvae.baseline import BaselineVQVAE
from src.networks.vqvae.baseline_2d import BaselineVQVAE2D
from src.networks.vqvae.single_vqvae import SingleVQVAE
from src.networks.vqvae.vqvae import VQVAEBase
from src.utils.general import get_max_decay_epochs
from src.utils.constants import VQVAENetworks, DecayWarmups


def get_vqvae_network(config: dict) -> VQVAEBase:
    if config["network"] == VQVAENetworks.SINGLE_VQVAE.value:
        network = SingleVQVAE(
            dimensions=config["data_spatial_dimension"],
            in_channels=config["data_num_channels"],
            out_channels=config["data_num_channels"],
            use_subpixel_conv=config["use_subpixel_conv"],
            use_slim_residual=config["use_slim_residual"],
            no_levels=config["no_levels"],
            downsample_parameters=config["downsample_parameters"],
            upsample_parameters=config["upsample_parameters"],
            no_res_layers=config["no_res_layers"],
            no_channels=config["no_channels"],
            codebook_type=config["codebook_type"],
            num_embeddings=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
            embedding_init=config["embedding_init"],
            commitment_cost=config["commitment_cost"],
            decay=config["decay"],
            norm=config["norm"],
            dropout=config["dropout"],
            act=config["act"],
            output_act=config["output_act"],
        )
    elif config["network"] == VQVAENetworks.BASELINE_VQVAE.value:
        if config["data_spatial_dimension"]==3:
            network = BaselineVQVAE(
                n_levels=config["no_levels"],
                downsample_parameters=config["downsample_parameters"],
                upsample_parameters=config["upsample_parameters"],
                n_embed=config["num_embeddings"][0],
                embed_dim=config["embedding_dim"][0],
                commitment_cost=config["commitment_cost"][0],
                n_channels=config["no_channels"],
                n_res_channels=config["no_channels"],
                n_res_layers=config["no_res_layers"],
                p_dropout=config["dropout"],
                vq_decay=config["decay"][0],
                use_subpixel_conv=config["use_subpixel_conv"],
                output_act=config["output_act"],
            )
        elif config["data_spatial_dimension"]==2:
            network = BaselineVQVAE2D(
                n_levels=config["no_levels"],
                downsample_parameters=config["downsample_parameters"],
                upsample_parameters=config["upsample_parameters"],
                n_embed=config["num_embeddings"][0],
                embed_dim=config["embedding_dim"][0],
                commitment_cost=config["commitment_cost"][0],
                n_channels=config["no_channels"],
                n_res_channels=config["no_channels"],
                n_res_layers=config["no_res_layers"],
                p_dropout=config["dropout"],
                vq_decay=config["decay"][0],
                use_subpixel_conv=config["use_subpixel_conv"],
                output_act=config["output_act"],
                n_input_channels=config["data_num_channels"]
            )
    elif config["network"] == VQVAENetworks.SLIM_VQVAE.value:
        raise NotImplementedError(
            f"{VQVAENetworks.SLIM_VQVAE}'s parsing is not implemented yet."
        )
    else:
        raise ValueError(
            f"VQVAE unknown. Was given {config['network']} but choices are {[vqvae.value for vqvae in VQVAENetworks]}."
        )

    return network


def add_vqvae_network_handlers(
    train_handlers: List, vqvae: VQVAEBase, config: dict, logger: Logger
) -> List:

    if type(config["decay_warmup"]) is str:
        config["decay_warmup"] = [config["decay_warmup"]] * len(config["decay"])

    assert len(config["decay_warmup"]) == len(config["decay"]), (
        f"decay_warmup and decay should have the same length, but got decay of length {len(config['decay'])} and "
        + f"decay_warmup of length {len(config['decay_warmup'])}"
    )

    if type(config["max_decay_epochs"]) is int:
        config["max_decay_epochs"] = [config["max_decay_epochs"]] * len(config["decay"])

    assert len(config["max_decay_epochs"]) == len(config["decay"]), (
        f"max_decay_epochs and decay should have the same length, but got decay of length {len(config['decay'])} and "
        + f"max_decay_epochs of length {len(config['max_decay_epochs'])}"
    )

    for index, decay_tuple in enumerate(
        zip(config["max_decay_epochs"], config["decay_warmup"], config["decay"])
    ):
        max_decay_epochs, decay_warmup, decay = decay_tuple

        if decay_warmup == DecayWarmups.STEP.value:
            delta_step = (0.99 - decay) / 4
            stair_steps = np.linspace(0, max_decay_epochs, 5)[1:]

            def decay_anealing(current_step: int) -> float:
                if (current_step + 1) >= stair_steps[3]:
                    return decay + 4 * delta_step
                if (current_step + 1) >= stair_steps[2]:
                    return decay + 3 * delta_step
                if (current_step + 1) >= stair_steps[1]:
                    return decay + 2 * delta_step
                if (current_step + 1) >= stair_steps[0]:
                    return decay + delta_step
                return decay

            train_handlers += [
                ParamSchedulerHandler(
                    parameter_setter=lambda x: vqvae.set_ema_decay(
                        decay=x, index=index
                    ),
                    value_calculator=decay_anealing,
                    vc_kwargs={},
                    epoch_level=True,
                )
            ]
        elif decay_warmup == DecayWarmups.LINEAR.value:
            train_handlers += [
                ParamSchedulerHandler(
                    parameter_setter=lambda x: vqvae.set_ema_decay(
                        decay=x, index=index
                    ),
                    value_calculator="linear",
                    vc_kwargs={
                        "initial_value": decay,
                        "step_constant": 0,
                        "step_max_value": max_decay_epochs
                        if isinstance(max_decay_epochs, int)
                        else get_max_decay_epochs(config=config, logger=logger),
                        "max_value": 0.99,
                    },
                    epoch_level=True,
                )
            ]
        elif decay_warmup == DecayWarmups.NONE.value:
            continue
        else:
            raise ValueError(
                f"Received decay_warmup value {decay_warmup} but the available choices are {[e.value for e in DecayWarmups]}"
            )

    return train_handlers
