import math
from enum import Enum
from typing import Union, Tuple

import numpy as np
import torch

from gilbert.gilbert2d import gilbert2d
from gilbert.gilbert3d import gilbert3d


class OrderingType(Enum):
    RASTER_SCAN = "raster_scan"
    S_CURVE = "s_curve"
    RANDOM = "random"
    HILBERT = "hilbert_curve"


class OrderingTransformations(Enum):
    ROTATE_90 = "rotate_90"
    TRANSPOSE = "transpose"
    REFLECT = "reflect"


class Ordering:
    def __init__(
        self,
        ordering_type: str,
        spatial_dims: int,
        dimensions: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        reflected_spatial_dims: Union[Tuple[bool, bool], Tuple[bool, bool, bool]],
        transpositions_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ],
        rot90_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ],
        transformation_order: Tuple[str, ...] = (
            OrderingTransformations.TRANSPOSE.value,
            OrderingTransformations.ROTATE_90.value,
            OrderingTransformations.REFLECT.value,
        ),
    ):
        super().__init__()
        self.ordering_type = ordering_type

        assert self.ordering_type in [
            e.value for e in OrderingType
        ], f"ordering_type must be one of the following {[e.value for e in OrderingType]}, but got {self.ordering_type}."

        self.spatial_dims = spatial_dims
        self.dimensions = dimensions

        assert (
            len(dimensions) == self.spatial_dims + 1
        ), f"Dimensions must have length {self.spatial_dims + 1}."

        self.reflected_spatial_dims = reflected_spatial_dims
        self.transpositions_axes = transpositions_axes
        self.rot90_axes = rot90_axes
        if len(set(transformation_order)) != len(transformation_order):
            raise ValueError(
                f"No duplicates are allowed. Received {transformation_order}."
            )

        for transformation in transformation_order:
            if transformation not in [t.value for t in OrderingTransformations]:
                raise ValueError(
                    f"Valid transformations are {[t.value for t in OrderingTransformations]} but received {transformation}."
                )
        self.transformation_order = transformation_order

        self.template = self._create_template()
        self._sequence_ordering = self._create_ordering()
        self._revert_sequence_ordering = np.argsort(self._sequence_ordering)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x[self._sequence_ordering]

        return x

    def get_sequence_ordering(self) -> np.ndarray:
        return self._sequence_ordering

    def get_revert_sequence_ordering(self) -> np.ndarray:
        return self._revert_sequence_ordering

    def _create_ordering(self):
        self.template = self._transform_template()
        order = self._order_template(template=self.template)

        return order

    def _create_template(self) -> np.ndarray:
        spatial_dimensions = self.dimensions[1:]
        template = np.arange(np.prod(spatial_dimensions)).reshape(*spatial_dimensions)

        return template

    def _transform_template(self) -> np.ndarray:
        for transformation in self.transformation_order:
            if transformation == OrderingTransformations.TRANSPOSE.value:
                self.template = self._transpose_template(template=self.template)
            elif transformation == OrderingTransformations.ROTATE_90.value:
                self.template = self._rot90_template(template=self.template)
            elif transformation == OrderingTransformations.REFLECT.value:
                self.template = self._flip_template(template=self.template)

        return self.template

    def _transpose_template(self, template: np.ndarray) -> np.ndarray:
        for axes in self.transpositions_axes:
            template = np.transpose(template, axes=axes)

        return template

    def _flip_template(self, template: np.ndarray) -> np.ndarray:
        for axis, to_reflect in enumerate(self.reflected_spatial_dims):
            template = np.flip(template, axis=axis) if to_reflect else template

        return template

    def _rot90_template(self, template: np.ndarray) -> np.ndarray:
        for axes in self.rot90_axes:
            template = np.rot90(template, axes=axes)

        return template

    def _order_template(self, template: np.ndarray) -> np.ndarray:
        depths = None
        if self.spatial_dims == 2:
            rows, columns = template.shape[0], template.shape[1]
        else:
            rows, columns, depths = (
                template.shape[0],
                template.shape[1],
                template.shape[2],
            )

        sequence = eval(f"self.{self.ordering_type}_idx")(rows, columns, depths)

        ordering = np.array([template[tuple(e)] for e in sequence])

        return ordering

    @staticmethod
    def raster_scan_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            for c in range(cols):
                if depths:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)

        return idx

    @staticmethod
    def s_curve_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            col_idx = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_idx:
                if depths:
                    depth_idx = (
                        range(depths) if c % 2 == 0 else range(depths - 1, -1, -1)
                    )

                    for d in depth_idx:
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)

        return idx

    @staticmethod
    def random_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        idx = []

        for r in range(rows):
            for c in range(cols):
                if depths:
                    for d in range(depths):
                        idx.append((r, c, d))
                else:
                    idx.append((r, c))

        idx = np.array(idx)
        np.random.shuffle(idx)

        return idx

    @staticmethod
    def hilbert_curve_idx(rows: int, cols: int, depths: int = None) -> np.ndarray:
        t = list(gilbert3d(rows, cols, depths) if depths else gilbert2d(rows, cols))
        idx = np.array(t)

        return idx


class RelativeSpatialPositioning(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        ordering: np.ndarray,
        dimensions: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        bucket_values: bool = False,
        bucket_beta: int = 50,
        conditioning_length: int = 0,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.dimensions = dimensions
        self.ordering = ordering
        self.conditioning_length = conditioning_length
        assert (
            len(dimensions) == self.spatial_dims + 1
        ), f"Dimensions must have length {self.spatial_dims + 1}."
        self.bucket_values = bucket_values
        self.bucket_beta = bucket_beta

        self.dist_array = self._get_distance_array()
        self.quantized_distances, self.num_buckets = self._rp_3d_product_and_quantize()

        self.ordered_distance_matrix = self.reorder()
        self.ordered_distance_matrix = self.account_conditionings()

    def get_pid_array(self):
        return self.ordered_distance_matrix

    def get_num_pids(self):
        return self.num_buckets

    def account_conditionings(self):
        if self.conditioning_length > 0:
            ordered_distance_matrix = (
                torch.ones(
                    self.ordered_distance_matrix.shape[0] + self.conditioning_length,
                    self.ordered_distance_matrix.shape[0] + self.conditioning_length,
                    dtype=self.ordered_distance_matrix.dtype,
                    device=self.ordered_distance_matrix.device,
                )
                * self.num_buckets
            )

            ordered_distance_matrix[
                self.conditioning_length :, self.conditioning_length :
            ] = self.ordered_distance_matrix

            self.num_buckets += 1

            return ordered_distance_matrix

        return self.ordered_distance_matrix

    def reorder(self):
        pid_rel_pos = self.quantized_distances.reshape(
            self.dimensions[1] * self.dimensions[2] * self.dimensions[3], -1
        )

        dim_1_reordered = torch.zeros_like(pid_rel_pos)
        for i in range(len(self.ordering)):
            dim_1_reordered[i] = pid_rel_pos[self.ordering[i]]

        dim_2_reordered = torch.zeros_like(pid_rel_pos)
        for i in range(len(self.ordering)):
            dim_2_reordered[:, i] = dim_1_reordered[:, self.ordering[i]]

        return dim_2_reordered

    def _get_distance_array(self):
        coord_array = torch.zeros(
            (self.dimensions[1], self.dimensions[2], self.dimensions[3], 3),
            dtype=torch.int,
        )
        height = coord_array.shape[0]
        width = coord_array.shape[1]
        depth = coord_array.shape[2]

        for i in range(height):
            for j in range(width):
                for k in range(depth):
                    coord_array[i, j, k, 0] = i
                    coord_array[i, j, k, 1] = j
                    coord_array[i, j, k, 2] = k

        dist_array = torch.zeros(
            (height, width, depth, height, width, depth, 3), dtype=torch.int
        )

        coord_array_heights = coord_array[:, :, :, 0]
        coord_array_widths = coord_array[:, :, :, 1]
        coord_array_depths = coord_array[:, :, :, 2]

        for i in range(height):
            for j in range(width):
                for k in range(depth):
                    dist_array[i, j, k, :, :, :, 0] = coord_array_heights - i
                    dist_array[i, j, k, :, :, :, 1] = coord_array_widths - j
                    dist_array[i, j, k, :, :, :, 2] = coord_array_depths - k

        return dist_array

    # Code adapted from iRPE in 2D:
    # https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DETR-with-iRPE/models/rpe_attention/irpe.py#L19
    def _rp_3d_product_and_quantize(self):

        alpha = self.bucket_beta / 2
        gamma = self.bucket_beta * 4

        if self.bucket_values:
            r = (
                self.piecewise_index(
                    self.dist_array[:, :, :, :, :, :, 0], alpha, self.bucket_beta, gamma
                )
                + self.bucket_beta
            )
            c = (
                self.piecewise_index(
                    self.dist_array[:, :, :, :, :, :, 1], alpha, self.bucket_beta, gamma
                )
                + self.bucket_beta
            )
            d = (
                self.piecewise_index(
                    self.dist_array[:, :, :, :, :, :, 2], alpha, self.bucket_beta, gamma
                )
                + self.bucket_beta
            )
        else:
            r = self.dist_array[:, :, :, :, :, :, 0]
            c = self.dist_array[:, :, :, :, :, :, 1]
            d = self.dist_array[:, :, :, :, :, :, 2]

        r = r - torch.min(r)
        c = c - torch.min(c)
        d = d - torch.min(d)

        max_dim = max(torch.max(r), torch.max(c), torch.max(d)) + 1

        pid = r + (c * max_dim) + (d * max_dim ** 2)

        return pid, torch.max(pid)

    @staticmethod
    def piecewise_index(relative_position, alpha, beta, gamma, dtype=torch.int):
        """piecewise index function

        Parameters
        ----------
        relative_position: torch.Tensor, dtype: long or float
            The shape of `relative_position` is (L, L).
        alpha, beta, gamma: float
            The coefficients of piecewise index function.
        Returns
        -------
        idx: torch.Tensor, dtype: long
            A tensor indexing relative distances to corresponding encodings.
            `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
        """
        rp_abs = relative_position.abs()
        mask = rp_abs <= alpha
        not_mask = ~mask
        rp_out = relative_position[not_mask]
        rp_abs_out = rp_abs[not_mask]
        y_out = (
            torch.sign(rp_out)
            * (
                alpha
                + torch.log(rp_abs_out / alpha)
                / math.log(gamma / alpha)
                * (beta - alpha)
            )
            .round()
            .clip(max=beta)
        ).to(dtype)

        idx = relative_position.clone()
        if idx.dtype in [torch.float32, torch.float64]:
            # round(x) when |x| <= alpha
            idx = idx.round().to(dtype)

        # assign the value when |x| > alpha
        idx[not_mask] = y_out
        return idx
