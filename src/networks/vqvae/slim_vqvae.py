import torch
import torch.nn as nn

from typing import List, Union, Sequence, Tuple, Dict

from monai.networks.blocks import ResidualUnit, MaxAvgPool, UpSample, Convolution
from monai.utils.enums import UpsampleMode

from src.networks.vqvae.vqvae import VQVAEBase
from src.layers.vector_quantization import VectorQuantizerEMA


class SlimVQVAE(VQVAEBase, nn.Module):
    """
    3D VQ-VAE 2 architecture loosely based on  [1].

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        act: activation type and arguments. Defaults to Leaky ReLU.
        norm: feature normalization type and arguments. Defaults to no norm.
        dropout: dropout ratio. Defaults to 0.1.
        adn_ordering: a string representing the ordering of activation, normalization,
            and dropout. Defaults to "NDA".
        no_levels: the amount of non-full-spatial-resolution the network has.
            Defaults to 4.
        level_zero_no_channels: the minimum channel dimension size. It gets multiplied by
            2**depth. Defaults to 4.
        no_res_layers: how many residual blocks are used on each level. Defaults to 1.
        use_subpixel_conv: Whether or not to use SubPixelConvolution for the last deconvolution. Defaults to False.
        codebook_type: VectorQuantization module type between "ema", "gradient", "relaxation". Defaults to "ema".
        quantizer_depths: at which depths of the network we have quantization blocks.
            Defaults to (2,3,4).
        quantizer_embedding_dims: to what channels size are the encodings and coditionings
            going to be bottlenecked before being fed to the quantization module. It also
            serves as the dimension of the quantization space. Defaults to (4,8,32).
        quantizer_num_embeddings: how many atomic elements will the dictionary of each
            quantization level have. Defaults to (8, 32, 256).
        quantizer_decays: the ema decay of each quantization level. Defaults to (0.99, 0.99, 0.99).
        quantizer_epsilon: the epsilons of each quantization level. Defaults to (1e-5, 1e-5, 1e-5).
        quantizer_commitment_costs: the commitment costs of each quantization level.
            Defaults to (0.25, 0.25, 0.25).
    References:
        [1] Tudosiu, P.D. el al 2020.
            Neuromorphologicaly-preserving Volumetric data encoding using VQ-VAE.
            arXiv preprint arXiv:2002.05692.
    """

    # < Python 3.9 TorchScript requirement for ModuleList
    __constants__ = ["encoder", "quantizer", "decoder"]

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        no_levels: int = 4,
        no_res_layers: int = 1,
        level_zero_no_channels: int = 4,
        use_subpixel_conv: bool = False,
        codebook_type: str = "ema",
        quantizer_depths: Tuple[int, ...] = (2, 3, 4),
        quantizer_num_embeddings: Tuple[int, ...] = (8, 32, 256),
        quantizer_embedding_dims: Tuple[int, ...] = (4, 8, 32),
        quantizer_commitment_costs: Tuple[float, ...] = (0.25, 0.25, 0.25),
        quantizer_decays: Tuple[float, ...] = (0.99, 0.99, 0.99),
        quantizer_epsilon: Tuple[float, ...] = (1e-5, 1e-5, 1e-5),
        adn_ordering: str = "NDA",
        norm: str = None,
        dropout: float = 0.1,
        act: str = "leakyrelu",
    ):
        super(SlimVQVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = dimensions

        self.downsample_kernel_size = 3
        self.downsample_stride = 2
        self.downsample_padding = 1

        self.upsample_mode = UpsampleMode.NONTRAINABLE

        self.act = act
        self.norm = norm
        # Casting to make sure TorchScript does not complain
        self.dropout = float(dropout)
        self.adn_ordering = adn_ordering

        self.no_levels = no_levels
        self.level_zero_no_channels = level_zero_no_channels
        self.no_res_layers = no_res_layers

        self.codebook_type = codebook_type
        self.quantizer_depths = quantizer_depths
        self.quantizer_num_embeddings = quantizer_num_embeddings
        self.quantizer_embedding_dims = quantizer_embedding_dims
        self.quantizer_commitment_costs = quantizer_commitment_costs
        self.quantizer_decays = quantizer_decays
        self.quantizer_epsilon = quantizer_epsilon

        self.__check_parameters()

        # TorchScript compatible initialization
        self.encoder: nn.ModuleList = self.construct_encoder()
        self.quantizer: nn.ModuleList = self.construct_quantizer()
        self.decoder: nn.ModuleList = self.construct_decoder()

    def get_ema_decay(self) -> Sequence[float]:
        ema_decay = [self.quantizer[1].get_ema_decay()]

        for idx in range(0, len(self.quantizer) - 2, 3):
            ema_decay.append(self.quantizer[idx + 4].get_ema_decay())

        return ema_decay

    def set_ema_decay(
        self, decay: Union[Sequence[float], float], index: int = None
    ) -> Sequence[float]:
        if index is not None:
            self.quantizer[1].set_ema_decay(
                decay if isinstance(decay, int) else decay[0]
            )

            i = 1
            for idx in range(0, len(self.quantizer) - 2, 3):
                self.quantizer[idx + 4].set_ema_decay(
                    decay if isinstance(decay, int) else decay[i]
                )
                i += 1
        elif type(index) is int and type(decay) is float:
            self.quantizer[index].set_ema_decay(decay)

        return self.get_ema_decay()

    def get_commitment_cost(self) -> Sequence[float]:
        commitment_cost = [self.quantizer[1].get_commitment_cost()]

        for idx in range(0, len(self.quantizer) - 2, 3):
            commitment_cost.append(self.quantizer[idx + 4].get_commitment_cost())

        return commitment_cost

    def set_commitment_cost(
        self, commitment_factor: Union[Sequence[float], float]
    ) -> Sequence[float]:
        self.quantizer[1].set_commitment_cost(
            commitment_factor
            if isinstance(commitment_factor, int)
            else commitment_factor[0]
        )

        i = 1
        for idx in range(0, len(self.quantizer) - 2, 3):
            self.quantizer[idx + 4].set_commitment_cost(
                commitment_factor
                if isinstance(commitment_factor, int)
                else commitment_factor[i]
            )
            i += 1

        return self.get_commitment_cost()

    def get_perplexity(self) -> Sequence[float]:
        perplexity = [self.quantizer[1].get_perplexity()]

        for idx in range(0, len(self.quantizer) - 2, 3):
            perplexity.append(self.quantizer[idx + 4].get_perplexity())

        return perplexity

    def __check_parameters(self) -> None:
        assert (
            max(self.quantizer_depths) <= self.no_levels
        ), f"The maximum depth for quantization ({max(self.quantizer_depths)}) is above the network's depth ({self.no_levels})"

        assert (
            len(self.quantizer_depths)
            == len(self.quantizer_embedding_dims)
            == len(self.quantizer_num_embeddings)
            == len(self.quantizer_decays)
            == len(self.quantizer_commitment_costs)
        ), (
            f"quantizer_depth ({len(self.quantizer_depths)}), "
            f"quantizer_embedding_dims ({len(self.quantizer_embedding_dims)}), "
            f"quantizer_atomic_elements ({len(self.quantizer_num_embeddings)}), "
            f"quantizer_decays ({len(self.quantizer_decays)}), "
            f"quantizer_commitment_costs ({len(self.quantizer_commitment_costs)}) "
            f"must have the same length."
        )

        assert (
            self.quantizer_embedding_dims[-1] >= 2 ** self.no_levels,
            f"The deepest quantization needs to have at least {2**self.no_levels} features, "
            f"but it has {self.quantizer_embedding_dims[-1]}",
        )

    def construct_encoder(self) -> nn.ModuleList:
        encoder = []

        current_depth = 0
        channels = self.level_zero_no_channels * 2 ** current_depth

        current_step = [
            ResidualUnit(
                dimensions=self.dimensions,
                in_channels=self.in_channels if idx == 0 else channels,
                out_channels=channels,
                adn_ordering=self.adn_ordering,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
            for idx in range(self.no_res_layers)
        ] + [
            MaxAvgPool(
                spatial_dims=self.dimensions,
                kernel_size=self.downsample_kernel_size,
                stride=self.downsample_stride,
                padding=self.downsample_padding,
            )
        ]

        for current_depth in range(1, self.no_levels + 1):
            channels = self.level_zero_no_channels * 2 ** current_depth
            current_step.extend(
                [
                    ResidualUnit(
                        dimensions=self.dimensions,
                        in_channels=channels,
                        out_channels=channels,
                        adn_ordering=self.adn_ordering,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                    )
                    for _ in range(self.no_res_layers)
                ]
            )

            if current_depth in self.quantizer_depths:
                encoder.append(nn.Sequential(*current_step))
                current_step = []

            current_step.append(
                MaxAvgPool(
                    spatial_dims=self.dimensions,
                    kernel_size=self.downsample_kernel_size,
                    stride=self.downsample_stride,
                    padding=self.downsample_padding,
                )
            )

        return nn.ModuleList(encoder)

    def construct_quantizer(self) -> nn.ModuleList:
        quantizer = []

        # Initialization to allow recurrent usage in the for loop.
        # It symbolises additional conditioning from quantization layers
        # bellow the current one.
        conditioning_channels = 0

        # We are building the quantization section from bottom to the top
        # due to the conditionings between levels
        for code_idx in reversed(range(len(self.quantizer_depths))):
            current_step = []

            # Preparing the quantization parameters
            depth = self.quantizer_depths[code_idx]
            embedding_dim = self.quantizer_embedding_dims[code_idx]
            num_embeddings = self.quantizer_num_embeddings[code_idx]
            decay = self.quantizer_decays[code_idx]
            commitment_cost = self.quantizer_commitment_costs[code_idx]
            epsilon = self.quantizer_epsilon[code_idx]

            channels = self.level_zero_no_channels * 2 ** depth

            current_step.extend(
                [
                    ResidualUnit(
                        dimensions=self.dimensions,
                        in_channels=channels
                        + (conditioning_channels if idx == 0 else 0),
                        out_channels=embedding_dim
                        if idx == self.no_res_layers - 1
                        else channels,
                        adn_ordering=self.adn_ordering,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                    )
                    for idx in range(self.no_res_layers)
                ]
            )
            quantizer.append(nn.Sequential(*current_step))

            if self.codebook_type == "ema":
                quantizer.append(
                    VectorQuantizerEMA(
                        dimensions=self.dimensions,
                        num_embeddings=num_embeddings,
                        embedding_dim=embedding_dim,
                        commitment_cost=commitment_cost,
                        decay=decay,
                        epsilon=epsilon,
                    )
                )
            elif self.codebook_type == "gradient":
                raise NotImplementedError(
                    "Gradient based codebooks not implemented yet."
                )
            elif self.codebook_type == "relaxation":
                raise NotImplementedError(
                    "Gumbel-softmax based codebooks not implemented yet."
                )
            else:
                raise NotImplementedError(
                    f"Available codebooks types are 'ema', 'gradient' and 'relaxation'. It was give {self.codebook_type}."
                )

            if code_idx > 0:
                current_step = []
                next_depth = self.quantizer_depths[code_idx - 1]

                conditioning_channels = embedding_dim

                for resolution_step in range(depth - next_depth):
                    current_step.append(
                        UpSample(
                            dimensions=self.dimensions,
                            in_channels=conditioning_channels,
                            out_channels=conditioning_channels // 2,
                            scale_factor=2,
                            mode=UpsampleMode.DECONV,
                        )
                    )

                    conditioning_channels = conditioning_channels // 2

                quantizer.append(nn.Sequential(*current_step))

        return nn.ModuleList(quantizer)

    def construct_decoder(self) -> nn.ModuleList:
        decoder = []

        # Local List copy of the Tuple
        quantizer_depths = list(self.quantizer_depths)

        # Initialization to allow recurrent usage in the for loop.
        in_channels = 0
        current_step = []

        for current_depth in range(self.no_levels, 0, -1):
            # Account for quantization skip connections
            quantization_channels = (
                self.quantizer_embedding_dims[quantizer_depths.index(current_depth)]
                if current_depth in quantizer_depths
                else 0
            )

            in_channels = in_channels + quantization_channels

            out_channels = (
                in_channels // 2
                if in_channels - quantization_channels > 0
                else quantization_channels // 2
            )

            current_step.extend(
                [
                    ResidualUnit(
                        dimensions=self.dimensions,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        adn_ordering=self.adn_ordering,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                    )
                    for _ in range(self.no_res_layers)
                ]
            )

            current_step.append(
                UpSample(
                    dimensions=self.dimensions,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=2,
                    mode=UpsampleMode.DECONV,
                )
            )

            in_channels = out_channels

            # If the output of all modules in current_step needs
            # to be concatenated with one of the quantizations
            # then we need to stop the nn.Sequential there
            if current_depth - 1 in quantizer_depths:
                decoder.append(nn.Sequential(*current_step))
                current_step = []

        current_step.extend(
            [
                ResidualUnit(
                    dimensions=self.dimensions,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    adn_ordering=self.adn_ordering,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                )
                for _ in range(self.no_res_layers)
            ]
        )

        current_step.append(
            Convolution(
                dimensions=self.dimensions,
                in_channels=out_channels,
                out_channels=self.out_channels,
                conv_only=True,
            )
        )

        decoder.append(nn.Sequential(*current_step))

        return nn.ModuleList(decoder)

    def get_last_layer(self) -> nn.parameter.Parameter:
        return list(self.decoder.modules())[-1].weight

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        encodings = []
        x = images

        for step in self.encoder:
            encodings.append(step(x))
            x = encodings[-1]

        return encodings

    def quantize(
        self, encodings: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Bellow we do the lowest level quantization which has an atypical step (does not have lower level
        # conditioning) so we have recurrent indexing in for loop

        # Squeezing
        x = self.quantizer[0](encodings[-1])

        # Quantization
        x_loss, x = self.quantizer[1](x)

        quantizations = [x]
        quantization_losses = [x_loss]

        for idx in range(0, len(encodings) - 1):
            # Reverse indexing due to TorchScript
            encoding_idx = -idx - 2
            step_idx = idx * 3 + 2

            # Upsampling
            x = self.quantizer[step_idx](x)

            # Concatenating
            x = torch.cat((x, encodings[encoding_idx]), 1)

            # Squeezing
            x = self.quantizer[step_idx + 1](x)
            # Quantization
            x_loss, x = self.quantizer[step_idx + 2](x)

            quantizations.append(x)
            quantization_losses.append(x_loss)

        return quantizations, quantization_losses

    def decode(self, quantizations: List[torch.Tensor]) -> torch.Tensor:
        x = self.decoder[0](quantizations[0])
        for i in range(1, len(self.decoder)):
            x = torch.cat((x, quantizations[i]), 1)
            x = self.decoder[i](x)
        return x

    def index_quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        encodings = self.encode(images=images)

        # Squeezing
        x = self.quantizer[0](encodings[-1])
        # Quantization
        _, _, e_idx = self.quantizer[1].quantize(x)
        _, x = self.quantizer[1](x)

        quantizations = [x]
        encoding_indices = [e_idx]

        for idx in range(0, len(encodings) - 1):
            # Reverse indexing due to TorchScript
            encoding_idx = -idx - 2
            step_idx = idx * 3 + 2

            # Upsampling
            x = self.quantizer[step_idx](x)

            # Concatenating
            x = torch.cat((x, encodings[encoding_idx]), 1)

            # Squeezing
            x = self.quantizer[step_idx + 1](x)
            # Quantization
            _, _, e_idx = self.quantizer[step_idx + 2].quantize(x)

            _, x = self.quantizer[step_idx + 2](x)

            quantizations.append(x)
            encoding_indices.append(e_idx)

        return encoding_indices

    def decode_samples(self, embedding_indices: List[torch.Tensor]) -> torch.Tensor:
        samples_codes = [self.quantizer[1].embed(embedding_indices[0])]
        for idx in range(0, len(embedding_indices) - 1):
            step_idx = idx * 3 + 2
            samples_codes.append(
                self.quantizer[step_idx + 2].embed(embedding_indices[idx + 1])
            )

        samples_images = self.decode(samples_codes)

        return samples_images

    def forward(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        encodings = self.encode(images)
        quantizations, quantization_losses = self.quantize(encodings)
        reconstruction = self.decode(quantizations)

        return {
            "reconstruction": [reconstruction],
            "quantization_losses": quantization_losses,
        }
