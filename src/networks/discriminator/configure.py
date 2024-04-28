import torch.nn as nn

from src.networks.discriminator.utils import DiscriminatorNetworks
from src.networks.discriminator.taming import TamingDiscriminator
from src.networks.discriminator.baseline import BaselineDiscriminator
from src.networks.discriminator.baseline_2d import BaselineDiscriminator2D


def get_discriminator_network(config: dict) -> nn.Module:

    if config["discriminator_network"] == DiscriminatorNetworks.BASELINE_DISCRIMINATOR.value and config["data_spatial_dimension"] ==3:
        network = BaselineDiscriminator(
            input_nc=config["data_num_channels"],
            ndf=config["discriminator_channels"],
            n_layers=config["discriminator_layers"],
        )
    elif config["discriminator_network"] == DiscriminatorNetworks.BASELINE_DISCRIMINATOR.value and config["data_spatial_dimension"] ==2:
        network = BaselineDiscriminator2D(
            input_nc=config["data_num_channels"],
            ndf=config["discriminator_channels"],
            n_layers=config["discriminator_layers"],
        )
    elif (
        config["discriminator_network"] == DiscriminatorNetworks.TAMING_DISCRIMINATOR.value
    ):
        network = TamingDiscriminator(
            dimensions=3,
            in_channels=1,
            no_channels=config["discriminator_channels"],
            no_layers=config["discriminator_layers"],
            act=config["discriminator_activation"],
            dropout=config["discriminator_dropout"],
            norm=config["discriminator_normalisation"]
        )
    else:
        raise ValueError(
            f"Discriminator unknown. Was given {config['discriminator_network']} but choices are"
            f" {[discriminator.value for discriminator in DiscriminatorNetworks]}."
        )

    return network


# If needs be one can implement an
#   def add_vqvae_network_handlers(
#        train_handlers: List, discriminator: nn.Module, config: dict, logger: Logger
#   ) -> List:
# to add discriminator specific handlers
