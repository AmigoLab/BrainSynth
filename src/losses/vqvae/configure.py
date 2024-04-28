from typing import Union, List

from torch.nn import Module
from torch.nn.modules.loss import _Loss

from src.handlers.general import ParamSchedulerHandler
from src.losses.vqvae.utils import VQVAELosses
from src.losses.vqvae.vqvae import *


def get_vqvae_loss(config: dict) -> Union[_Loss, Module]:
    """
    Configures and returns the loss.

    Expects a 'loss' field in the dictionary which can have any of the vlaues found in src.losses.configure.VQVAELosses.
    """
    if config["loss"] == VQVAELosses.BAUR.value:
        loss = BaurLoss()
    elif config["loss"] == VQVAELosses.MSE.value:
        loss = MSELoss()
    elif config["loss"] == VQVAELosses.SPECTRAL.value:
        loss = SpectralLoss(dimensions=config["data_spatial_dimension"])
    elif config["loss"] == VQVAELosses.HARTLEY.value:
        loss = HartleyLoss(dimensions=config["data_spatial_dimension"])
    elif config["loss"] == VQVAELosses.JUKEBOX.value:
        loss = JukeboxLoss(dimensions=config["data_spatial_dimension"])
    elif config["loss"] == VQVAELosses.WAVEGAN.value:
        loss = WaveGANLoss(dimensions=config["data_spatial_dimension"])
    elif config["loss"] == VQVAELosses.PERCEPTUAL.value:
        loss = PerceptualLoss(
            dimensions=config["data_spatial_dimension"], drop_ratio=0.50
        )
    elif config["loss"] == VQVAELosses.JUKEBOX_PERCEPTUAL.value:
        loss = JukeboxPerceptualLoss(
            dimensions=config["data_spatial_dimension"], drop_ratio=0.5
        )
    elif config["loss"] == VQVAELosses.HARTLEY_PERCEPTUAL.value:
        loss = HartleyPerceptualLoss(
            dimensions=config["data_spatial_dimension"], drop_ratio=0.5
        )
    elif config["loss"] == VQVAELosses.JUKEBOX_PERCEPTUAL_SSIM.value:
        loss = JukeboxPerceptualSSIMLoss(
            dimensions=3,
            drop_ratio=0.5,
            input_range=JukeboxPerceptualSSIMLoss.input_ranges[1]
            if config["output_act"] == "TANH"
            else JukeboxPerceptualSSIMLoss.input_ranges[0],
            ssim_win_size=config["ssim_win_size"],
        )
    elif config["loss"] == VQVAELosses.SSIM.value:
        loss = SSIMLoss(dimensions=3, ssim_win_size=config["ssim_win_size"])
    elif config["loss"] == VQVAELosses.SSIM_HARTLEY.value:
        loss = SSIMHartleyLoss(dimensions=3, ssim_win_size=config["ssim_win_size"])
    elif config["loss"] == VQVAELosses.SSIM_MED3D.value:
        loss = SSIMMed3DLoss(
            dimensions=3,
            med3d_checkpoint_path=config["med3d_checkpoint_path"],
            med3d_depth=config["med3d_loss_depth"],
            ssim_win_size=config["ssim_win_size"],
        )
    elif config["loss"] == VQVAELosses.BASELINE.value:
        loss = BaselineLoss()
    else:
        raise ValueError(
            f"Loss function unknown. Was given {config['loss']} but choices are {[loss.value for loss in VQVAELosses]}."
        )

    return loss


def add_vqvae_loss_handlers(
    train_handlers: List, loss_function: Union[_Loss, Module], config: dict
) -> List:
    """
    Configures the required handlers for each loss. Please see implementation for details.
    """
    if config["loss"] == "baur":
        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=loss_function.set_gdl_factor,
                value_calculator="linear",
                vc_kwargs={
                    "initial_value": config["initial_factor_value"],
                    "step_constant": config["initial_factor_steps"],
                    "step_max_value": config["max_factor_steps"],
                    "max_value": config["max_factor_value"],
                },
                epoch_level=True,
            )
        ]
    return train_handlers
