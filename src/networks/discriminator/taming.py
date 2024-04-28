import torch.nn as nn

from monai.networks.blocks import Convolution
from monai.networks.layers import Norm


class TamingDiscriminator(nn.Module):
    """
    MONAI based implementation of the PatchGAN Discriminator in [1] which is based on [2].

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.

    References:
        [1] Esser, P., Rombach, R. and Ommer, B., 2020.
        Taming Transformers for High-Resolution Image Synthesis.
        arXiv preprint arXiv:2012.09841.
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py#L17

        [2] Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2017.
        Image-to-image translation with conditional adversarial networks.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int = 1,
        no_channels: int = 64,
        no_layers: int = 3,
        act: str = "LEAKYRELU",
        dropout: float = 0.0,
        norm: Norm = "BATCH",
    ):
        super(TamingDiscriminator, self).__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.act = act
        self.dropout = dropout
        self.norm = norm
        self.no_channels = no_channels
        self.no_layers = no_layers

        self.bias = str.upper(self.norm) != "BATCH"
        self.kernel_size = 4
        self.padding = 1

        sequence = [
            Convolution(
                dimensions=self.dimensions,
                in_channels=self.in_channels,
                out_channels=self.no_channels,
                kernel_size=self.kernel_size,
                strides=2,
                padding=self.padding,
                act=self.act,
                dropout=0.0,
                norm=None,
                bias=True,
            )
        ]

        out_channels_multiplier = 1
        for i in range(1, self.no_layers):
            in_channels_multiplier = out_channels_multiplier
            out_channels_multiplier = min(2 ** i, 8)

            sequence += [
                Convolution(
                    dimensions=self.dimensions,
                    in_channels=self.no_channels * in_channels_multiplier,
                    out_channels=self.no_channels * out_channels_multiplier,
                    kernel_size=self.kernel_size,
                    strides=2,
                    padding=self.padding,
                    bias=self.bias,
                    act=self.act,
                    dropout=self.dropout,
                    norm=self.norm,
                )
            ]

        in_channels_multiplier = out_channels_multiplier
        out_channels_multiplier = min(2 ** self.no_layers, 8)

        sequence += [
            Convolution(
                dimensions=self.dimensions,
                in_channels=self.no_channels * in_channels_multiplier,
                out_channels=self.no_channels * out_channels_multiplier,
                kernel_size=self.kernel_size,
                strides=1,
                padding=self.padding,
                bias=self.bias,
                act=self.act,
                dropout=self.dropout,
                norm=self.norm,
            )
        ]

        sequence += [
            Convolution(
                dimensions=self.dimensions,
                in_channels=self.no_channels * out_channels_multiplier,
                out_channels=1,
                kernel_size=self.kernel_size,
                strides=1,
                padding=self.padding,
                bias=True,
                act=None,
                dropout=None,
                norm=None,
                conv_only=True,
            )
        ]

        self.net = nn.Sequential(*sequence)

        self.apply(self.weights_init)

    def forward(self, input):
        return self.net(input)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        # This is done to avoid calling .weight.data on the MONAI Convolution which does not have one
        if classname.find("Conv") != -1 and classname.find("Convolution") == -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
