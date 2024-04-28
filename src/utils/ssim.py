import warnings
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class SSIM(torch.nn.Module):
    """
        Args:
            spatial_dims (int): Number of spatial dimensions, could be 2, or 3.
            in_channels (int): Number of input channels.
            data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
                Defaults to 255.0.
            size_average (bool): If size_average=True, SSIM of all images will be averaged as a scalar.
            gaussian_kernel (torch.Tensor): 1-D gaussian kernel. If None, a new kernel will be created according to
                gaussian_kernel_size and gaussian_kernel_sigma. Defaults to None.
            gaussian_kernel_size (int): Size of the gaussian kernel. Defaults to 11 from Ref 1 - Sec 3.C.
            gaussian_kernel_sigma (float): Sigma of gaussian distribution. Defaults to 1.5 from Ref 1 - Sec 3.C.
            k (Tuple[float,float]): Scalar constants (k1, k2). Try a larger k2 constant (e.g. 0.4) if you get a
                negative or NaN results. Defaults to (0.01, 0.03) from Ref 1 - Sec 3.C.
            scales (Tuple[float,float,float]): Scales of the luminance, contrast and structure components of the SSIM.
                Defaults to (1.0,1.0,1.0) from Ref 1 - Sec 3.B.
            gradient_based (bool): Whether or not the structural and contrast components of the SSIM are based on the
                gradient magnitudes of the images as per Ref 3. Defaults to True.
            star_based (bool): Whether or not to use alternative mathematical stability from Ref 4. Defaults to True.
            gradient_masks_weights (Union[Tuple[float,float,float],Tuple[float,float,float,float]]): The weight of the
                gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as
                per Ref 4 - Sec 3.2 and the default values would be (0.5, 0.25, 0.25). Or 4 as per Ref 5 - Sec 3.2 and
                the default values would be (0.25, 0.25, 0.25, 0.25). If it is None no gradient masks will be applied.
                Defaults to (0.25, 0.25, 0.25, 0.25).
            multi_scale_weights (Tuple[float,float,float,float]): The weights for each scale of the Multi-Scale SSIM.
                Defaults to (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) as per Ref 2 - Sec 3.2.

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.

            [2] Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November.
            Multiscale structural similarity for image quality assessment.
            In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.

            [3] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
            Gradient-based structural similarity for image quality assessment.
            In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

            [4] Li, C. and Bovik, A.C., 2009, January.
            Three-component weighted structural similarity index.
            In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

            [5] Li, C. and Bovik, A.C., 2010.
            Content-partitioned structural similarity index for image quality assessment.
            Signal Processing: Image Communication, 25(7), pp.517-526.

            [6] Rouse, D.M. and Hemami, S.S., 2008, February.
            Analyzing the role of visual structure in the recognition of natural image content with multi-scale SSIM.
            In Human Vision and Electronic Imaging XIII (Vol. 6806, p. 680615).
            International Society for Optics and Photonics.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        data_range: float = 255.0,
        reduction: str = "mean",
        size_average: bool = True,
        gaussian_kernel: torch.Tensor = None,
        gaussian_kernel_size: int = 11,
        gaussian_kernel_sigma: float = 1.5,
        k: Tuple[float, float] = (0.01, 0.03),
        scales: Tuple[float, float, float] = (1, 1, 1),
        multi_scale_weights: Tuple[float, ...] = (
            0.0448,
            0.2856,
            0.3001,
            0.2363,
            0.1333,
        ),
        gradient_based: bool = True,
        gradient_masks_weights: Union[
            Tuple[float, float, float], Tuple[float, float, float, float]
        ] = (0.25, 0.25, 0.25, 0.25),
        star_based: bool = True,
    ):
        super(SSIM, self).__init__()
        self.spatial_dims = spatial_dims

        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"Input images should be either 2D or 3D, but got {self.spatial_dims}"
            )

        self.conv = F.conv2d if spatial_dims == 2 else F.conv3d
        self.avg_pool = F.avg_pool2d if spatial_dims == 2 else F.avg_pool3d
        self.in_channels = in_channels
        self.data_range = data_range
        self.size_average = size_average

        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_kernel_sigma = gaussian_kernel_sigma

        if gaussian_kernel:
            self.register_buffer(name="gaussian_kernel", tensor=gaussian_kernel)
            self.gaussian_kernel_size = self.gaussian_kernel.shape[-1]
        else:
            self.register_buffer(
                name="gaussian_kernel", tensor=self._fspecial_gauss_1d()
            )

        if self.gaussian_kernel_size % 2 != 1:
            raise ValueError(
                f"Gaussian kernel size should be odd, but got {self.gaussian_kernel_size}."
            )

        # TODO Clarify the assert
        assert all(
            [ws == 1 for ws in self.gaussian_kernel.shape[1:-1]]
        ), self.gaussian_kernel_size.shape

        # Here we calculate by how much the SSIM map differs from input image shape. This is calculated based on pytorch
        # conv documentation and the fact the stride=1, padding=0 and dilation=1 during all convolutions
        self.shape_delta = self.gaussian_kernel_size - 1

        self.k = k
        self.scales = scales
        if multi_scale_weights:
            self.register_buffer(
                name="multi_scale_weights", tensor=torch.tensor(multi_scale_weights)
            )
        else:
            self.multi_scale_weights = None

        # We prepare the sobel kernels for the gradient calculations of the images from which the new contrast and
        # structure SSIM components will be calculated [3] or for the calculations of the masks [4] [5]
        for idx, sobel_kernel in enumerate(self._sobel_kernels()):
            self.register_buffer(name=f"gradient_kernel_{idx}", tensor=sobel_kernel)

        self.gradient_based = gradient_based
        self.gradient_masks_weights = gradient_masks_weights

        self.star_based = star_based
        self.register_buffer(name="star_threshold", tensor=torch.zeros(()))

    def _prepare_weight(self, kernel_weights: torch.Tensor) -> torch.Tensor:
        # TODO: Clean it up by using reshape
        k = kernel_weights.unsqueeze(0).unsqueeze(0)
        k = k.repeat([self.in_channels] + [1] * (self.spatial_dims + 1))
        return k

    def _fspecial_gauss_1d(self) -> torch.Tensor:
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = torch.arange(self.gaussian_kernel_size).to(dtype=torch.float)
        coords -= self.gaussian_kernel_size // 2

        g = torch.exp(-(coords ** 2) / (2 * self.gaussian_kernel_sigma ** 2))
        g /= g.sum()

        return self._prepare_weight(kernel_weights=g)

    def _sobel_kernels(self):
        # TODO: Make the other kernels be based on the x kernel and use transpose to generate them
        # The kernels are pre-normalised for cleaner code
        if self.spatial_dims == 2:
            g_x = self._prepare_weight(
                kernel_weights=torch.tensor(
                    [[-0.125, 0, +0.125], [-0.25, 0, +0.25], [-0.125, 0, +0.125]]
                )
            )

            g_y = self._prepare_weight(
                torch.tensor(
                    [[-0.125, -0.25, -0.125], [0, 0, 0], [+0.125, +0.25, +0.125]]
                )
            )

            return g_x, g_y
        elif self.spatial_dims == 3:
            g_x = self._prepare_weight(
                torch.tensor(
                    [
                        [[-0.05, 0, +0.05], [-0.05, 0, +0.05], [-0.05, 0, +0.05]],
                        [[-0.05, 0, +0.05], [-0.1, 0, +0.1], [-0.05, 0, +0.05]],
                        [[-0.05, 0, +0.05], [-0.05, 0, +0.05], [-0.05, 0, +0.05]],
                    ]
                )
            )

            g_y = self._prepare_weight(
                torch.tensor(
                    [
                        [[-0.05, -0.05, -0.05], [0, 0, 0], [+0.05, +0.05, +0.05]],
                        [[-0.05, -0.1, -0.05], [0, 0, 0], [+0.05, +0.1, +0.05]],
                        [[-0.05, -0.05, -0.05], [0, 0, 0], [+0.05, +0.05, +0.05]],
                    ]
                )
            )

            g_z = self._prepare_weight(
                torch.tensor(
                    [
                        [
                            [-0.05, -0.05, -0.05],
                            [-0.05, -0.1, -0.05],
                            [-0.05, -0.05, -0.05],
                        ],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [
                            [+0.05, +0.05, +0.05],
                            [+0.05, +0.1, +0.05],
                            [+0.05, +0.05, +0.05],
                        ],
                    ]
                )
            )

            return g_x, g_y, g_z

    def gradient_map(self, input: torch.Tensor) -> torch.Tensor:
        """
        Calculating the gradients of the images based on the gradient definition in [1]

        Args:
            input (torch.Tensor): Batch of images

        References:
            [1] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
            Gradient-based structural similarity for image quality assessment.
            In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.
        """
        gradient = torch.zeros_like(input)

        if self.spatial_dims == 2:
            gradient = gradient[:, :, 1:-1, 1:-1]
        elif self.spatial_dims == 3:
            gradient = gradient[:, :, 1:-1, 1:-1, 1:-1]

        # We are following the gradient magnitude definition from Ref 1 - Section 3.1 - Eq 1
        for idx in range(self.spatial_dims):
            if idx == 0:
                gradient_kernel = self.gradient_kernel_0
            elif idx == 1:
                gradient_kernel = self.gradient_kernel_1
            elif idx == 2:
                gradient_kernel = self.gradient_kernel_2

            directional_gradient = torch.abs(
                self.conv(
                    input=input,
                    weight=gradient_kernel,
                    stride=1,
                    padding=0,
                    groups=input.shape[1],
                    dilation=1,
                )
            )
            gradient = gradient + directional_gradient

        return gradient

    def get_gradient_masks(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Calculating the masks based on the gradient images of x and y based on [1] and [2].

        Args:
            x (torch.Tensor): A batch of images
            y (torch.Tensor): A batch of images

        Return :
             Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
             Masks based on gradients

        References:
            [1] Li, C. and Bovik, A.C., 2009, January.
            Three-component weighted structural similarity index.
            In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

            [2] Li, C. and Bovik, A.C., 2010.
            Content-partitioned structural similarity index for image quality assessment.
            Signal Processing: Image Communication, 25(7), pp.517-526.
        """
        x_g = self.gradient_map(input=x)
        y_g = self.gradient_map(input=y)

        g_max, _ = torch.max(torch.flatten(x_g, start_dim=2), -1)
        th1 = 0.12 * g_max
        th1 = torch.reshape(th1, shape=th1.shape + (1,) * self.spatial_dims)
        th2 = 0.06 * g_max
        th2 = torch.reshape(th2, shape=th2.shape + (1,) * self.spatial_dims)

        if len(self.gradient_masks_weights) == 3:
            # Ref 1 - Sec 3.1 - Step 3 - R 1
            m1 = torch.logical_and(x_g > th1, y_g > th1)

            # Ref 1 - Sec 3.1 - Step 3 - R 2
            m2 = torch.logical_and(x_g < th2, y_g <= th1)

            # Ref 1 - Sec 3.1 - Step 3 - R 3
            m3 = torch.logical_not(torch.logical_or(m1, m2))

            gradient_masks = (m1.float(), m2.float(), m3.float())
        elif len(self.gradient_masks_weights) == 4:
            # The Masks calculation the are following are not calculated based on the actual publication, but on the
            # source code that was obtained from the lead author.

            # Ref 2 - Sec 3.1 - Step 3 - R 1
            m1 = torch.logical_and(x_g > th1, y_g > th1)

            # Ref 2 - Sec 3.1 - Step 3 - R 2
            m2 = torch.logical_or(x_g > th1, y_g > th1)
            m2 = torch.logical_and(m2, torch.logical_not(m1))

            # Ref 2 - Sec 3.1 - Step 3 - R 3
            m3 = torch.logical_and(x_g < th2, y_g <= th1)
            m3 = torch.logical_and(m3, torch.logical_not(torch.logical_or(m1, m2)))

            # Ref 2 - Sec 3.1 - Step 3 - R 4
            m4 = torch.logical_not(torch.logical_or(torch.logical_or(m1, m2), m3))

            gradient_masks = (m1.float(), m2.float(), m3.float(), m4.float())
        # Cropping the gradient masks to fit the post smoothing images, this enable us to have calculations only in
        # convolutionally valid points
        if self.spatial_dims == 2:
            gradient_masks = tuple(
                gradient_mask[
                    :,
                    :,
                    self.shape_delta // 2 : -self.shape_delta // 2,
                    self.shape_delta // 2 : -self.shape_delta // 2,
                ]
                for gradient_mask in gradient_masks
            )
        elif self.spatial_dims == 3:
            gradient_masks = tuple(
                gradient_mask[
                    :,
                    :,
                    self.shape_delta // 2 : -self.shape_delta // 2,
                    self.shape_delta // 2 : -self.shape_delta // 2,
                    self.shape_delta // 2 : -self.shape_delta // 2,
                ]
                for gradient_mask in gradient_masks
            )

        return gradient_masks

    def _gaussian_filter(self, input):
        """
        Blur input with 1-D kernel

        Args:
            input (torch.Tensor): a batch of tensors to be blurred

        Returns:
            torch.Tensor: blurred tensors
        """
        out = input
        for i, s in enumerate(input.shape[2:]):
            if s >= self.gaussian_kernel.shape[-1]:
                out = self.conv(
                    input=out,
                    weight=self.gaussian_kernel.transpose(2 + i, -1),
                    padding=0,
                    stride=1,
                    groups=self.in_channels,
                    dilation=1,
                )
            else:
                warnings.warn(
                    f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {self.gaussian_kernel.shape[-1]}"
                )

        return out

    def _ssim(self, original_images: torch.Tensor, perturbed_images: torch.Tensor):
        """
        Calculate ssim index for X and Y

        Args:
            original_images (torch.Tensor): A batch of images, (N,C,[T,]H,W).
            perturbed_images (torch.Tensor): A batch of images, (N,C,[T,]H,W).
            data_range (float): Maximum value of the input images assuming 0 is the minimum. Usually 1.0 or 255.0.
                Defaults to 255.0.
            win (torch.Tensor): A 1-D Gaussian kernel.
            k (Tuple[float,float]): Scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a
                negative or NaN results. Defaults to (0.01, 0.03) from Ref 1 - Sec 3.C.
            scales (Tuple[float,float,float]): Scales of the luminance, contrast and structure components of the SSIM.
                Defaults to (1.0,1.0,1.0) from Ref 1 - Sec 3.B.
            gradient_masks (Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],]):
                Gradient masks that are determined based on the gradient magnitude of the images as per [3] or [4].
                Defaults to None.
            gradient_masks_weights (Union[Tuple[float, float, float], Tuple[float, float, float, float]]): The weight of the
                gradient masked regions. It also dictates the number of masks that will be used. It can be either 3 as per
                [3] and the default values would be (0.5, 0.25, 0.25). Or 4 as per [4] and the default values would be
                (0.25, 0.25, 0.25, 0.25). If it is None no gradient masks will be applied. Defaults to None.
            gradient_kernels (Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
                The gradient kernels that will be used to calculate the gradient magnitudes of each image. Defaults to None.
        Returns:
            torch.Tensor: SSIM results.

        References:
            [1] Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004.
            Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), pp.600-612.

            [2] Chen, G.H., Yang, C.L. and Xie, S.L., 2006, October.
            Gradient-based structural similarity for image quality assessment.
            In 2006 International Conference on Image Processing (pp. 2929-2932). IEEE.

            [3] Li, C. and Bovik, A.C., 2009, January.
            Three-component weighted structural similarity index.
            In Image quality and system performance VI (Vol. 7242, p. 72420Q). International Society for Optics and Photonics.

            [4] Li, C. and Bovik, A.C., 2010.
            Content-partitioned structural similarity index for image quality assessment.
            Signal Processing: Image Communication, 25(7), pp.517-526.

            [5] Rouse, D.M. and Hemami, S.S., 2008, February.
            Analyzing the role of visual structure in the recognition of natural image content with multi-scale SSIM.
            In Human Vision and Electronic Imaging XIII (Vol. 6806, p. 680615).
            International Society for Optics and Photonics.
        """
        K1, K2 = self.k
        alpha, beta, gamma = self.scales

        C1 = (K1 * self.data_range) ** 2
        C2 = (K2 * self.data_range) ** 2
        C3 = C2 / 2

        # TODO: Replace this with fftconvolution to follow the original implementation from [1]
        mu1 = self._gaussian_filter(original_images)
        mu2 = self._gaussian_filter(perturbed_images)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Ref 1 - Sec 3.B - Eq 6
        luminance_numerator = 2 * mu1_mu2 + C1
        luminance_denominator = mu1_sq + mu2_sq + C1

        # Ref 5 - Sec 5.3.3 - Eq 6
        if self.star_based:
            denominator_mask = torch.logical_not(
                torch.logical_or(
                    luminance_denominator < self.star_threshold,
                    luminance_denominator > self.star_threshold,
                )
            ).float()

            luminance_denominator = (
                luminance_denominator + luminance_numerator * denominator_mask
            )

        luminance = luminance_numerator / luminance_denominator
        # Calculating the gradient masks for [3] or [4]
        if self.gradient_masks_weights:
            gradient_masks = self.get_gradient_masks(original_images, perturbed_images)

        # Calculating the gradients for [2]
        if self.gradient_based:
            # The luminance is cropped to fit the structure and contrast of gradient based images. This guarantees that
            # all of our measurements are done in areas in which the metric is defined.
            if self.spatial_dims == 2:
                luminance = luminance[:, :, 1:-1, 1:-1]
            else:
                luminance = luminance[:, :, 1:-1, 1:-1, 1:-1]

            original_images = self.gradient_map(input=original_images)
            perturbed_images = self.gradient_map(input=perturbed_images)

            # Redefining variables since they will be used in the structure and contrast calculations
            mu1 = self._gaussian_filter(original_images)
            mu2 = self._gaussian_filter(perturbed_images)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

        sigma1_sq = self._gaussian_filter(original_images * original_images) - mu1_sq
        sigma2_sq = self._gaussian_filter(perturbed_images * perturbed_images) - mu2_sq

        sigma12 = self._gaussian_filter(original_images * perturbed_images) - mu1_mu2

        # Here we add an epsilon to compensate for numerical precision issues to make sure we do not get NaNs from the
        # sqrt calculations.
        sigma1 = torch.sqrt(
            sigma1_sq
            + 1e-5
            * (self._gaussian_filter(original_images * original_images) + mu1_sq)
            / 2
        )
        sigma2 = torch.sqrt(
            sigma2_sq
            + 1e-5
            * (self._gaussian_filter(perturbed_images * perturbed_images) + mu2_sq)
            / 2
        )

        contrast_numerator = 2 * sigma1 * sigma2 + C2
        contrast_denominator = sigma1_sq + sigma2_sq + C2

        # Ref 5 - Sec 5.3.3 - Eq 7
        if self.star_based:
            denominator_mask = torch.logical_not(
                torch.logical_or(
                    contrast_denominator < self.star_threshold,
                    contrast_denominator > self.star_threshold
                )
            )

            contrast_denominator = (
                contrast_denominator + contrast_numerator * denominator_mask
            )

        # Ref 1 - Sec 3.B - Eq 9
        contrast = contrast_numerator / contrast_denominator

        # Ref 1 - Sec 3.B - Eq 10
        structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

        # Ref 5 - Sec 5.3.3 - Eq 8
        if self.star_based:
            mask = torch.logical_not(
                torch.logical_or(
                    torch.logical_and(sigma1 > sigma2, sigma2 == 0),
                    torch.logical_and(sigma2 > sigma1, sigma1 == 0),
                )
            )

            structure = structure * mask

            mask = torch.logical_and(sigma1 == 0, sigma2 == 0)

            structure = structure * torch.logical_not(mask).float() + mask.float()

        # Ref 1 - Sec 3.B - Eq 12
        luminance = torch.pow(luminance, alpha)
        contrast = torch.pow(contrast, beta)
        structure = torch.pow(structure, gamma)

        ssim_map = luminance * contrast * structure
        # We also output the contrast and structure map for MS-SSIM calculations
        cs_map = contrast * structure

        if self.gradient_masks_weights:
            if not self.gradient_based:
                if self.spatial_dims == 2:
                    ssim_map = ssim_map[:, :, 1:-1, 1:-1]
                    cs_map = cs_map[:, :, 1:-1, 1:-1]

                else:
                    ssim_map = ssim_map[:, :, 1:-1, 1:-1, 1:-1]
                    cs_map = cs_map[:, :, 1:-1, 1:-1, 1:-1]

            # Rebalancing the weights in case a mask is full of zeros.
            # It happens at low resolution during the multi scales calculation.

            # Masks that are used during the calculations
            usable_masks = np.array([(gm == 1).any().cpu().numpy() for gm in gradient_masks])
            # Masks that are not used during the calculations
            unusable_masks = np.array([not gm for gm in usable_masks])
            gradient_masks_weights = np.array(self.gradient_masks_weights)
            # Summing up the weights of the unused masks
            to_be_redistributed = sum(gradient_masks_weights[unusable_masks])
            # Counting the masks that are used
            usable_count = sum(usable_masks)
            if usable_count < len(usable_masks):
                # Setting to zero the unused masks
                gradient_masks_weights[unusable_masks] = 0
                # Redistributing the unused weights
                gradient_masks_weights[usable_masks] += (
                    to_be_redistributed / usable_count
                )
                # Rescaling the weights to sum up to 1
                gradient_masks_weights /= sum(gradient_masks_weights)

            ssim = torch.zeros_like(torch.flatten(ssim_map, start_dim=2).mean(-1))

            cs = torch.zeros_like(torch.flatten(ssim_map, start_dim=2).mean(-1))

            # TODO: Apply round with tolerance when available in PyTorch so in the cases of 0.33(3) weights we end up with
            #   a total ssim/cs of 1
            for gm, gmw in zip(gradient_masks, gradient_masks_weights):
                value = torch.flatten(input=ssim_map * gm, start_dim=2).sum(-1)
                value /= torch.flatten(input=gm, start_dim=2).sum(-1) + 1e-5
                value *= gmw
                ssim += value

                value = torch.flatten(input=cs_map * gm, start_dim=2).sum(-1)
                value /= torch.flatten(input=gm, start_dim=2).sum(-1) + 1e-5
                value *= gmw
                cs += value
        else:
            ssim = torch.flatten(ssim_map, start_dim=2).mean(-1)
            cs = torch.flatten(cs_map, start_dim=2).mean(-1)

        ssim = torch.torch.relu(ssim)
        cs = torch.relu(cs)

        return ssim, cs

    def _ms_ssim(self, original_images: torch.Tensor, perturbed_images: torch.Tensor):
        divisible_by = 2 ** (len(self.multi_scale_weights) - 1)
        bigger_than = (
            self.gaussian_kernel_size + 2 if self.gradient_based else 0
        ) * 2 ** (len(self.multi_scale_weights) - 1)

        for idx, shape_size in enumerate(original_images.shape[2:]):
            assert (
                shape_size % divisible_by == 0
            ), f"Image size needs to be divisible by {divisible_by} but dimension {idx + 2} has size {shape_size}"

            assert (
                shape_size >= bigger_than
            ), f"Image size needs to be higher than {bigger_than} but dimension {idx + 2} has size {shape_size}"

        levels = self.multi_scale_weights.shape[0]
        mscs = []

        for i in range(levels):
            ssim, cs = self._ssim(original_images, perturbed_images)
            if i < levels - 1:
                mscs.append(cs)
                padding = [s % 2 for s in original_images.shape[2:]]
                original_images = self.avg_pool(
                    original_images, kernel_size=2, padding=padding
                )
                perturbed_images = self.avg_pool(
                    perturbed_images, kernel_size=2, padding=padding
                )

        mscs_and_ssim = torch.stack(mscs + [ssim], dim=0)
        msssim = torch.prod(
            mscs_and_ssim ** self.multi_scale_weights.view(-1, 1, 1), dim=0
        )

        return msssim

    def forward(self, original_image, perturbed_image):
        original_image = original_image.float()
        perturbed_image = perturbed_image.float()

        if self.multi_scale_weights is not None:
            metric = self._ms_ssim(original_image, perturbed_image)
        else:
            metric, _ = self._ssim(original_image, perturbed_image)

        return metric
