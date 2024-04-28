from collections import OrderedDict
from typing import Collection, Dict, Hashable, Mapping, Union

import numpy as np
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.misc import ensure_tuple


class TraceTransformsd(MapTransform):
    """
    This class is a quick and dirty component for Distribution Augmentation paper.
    Should not be heavily be relied uppon and in the future it should be modified to take advantage of the
        traceable transformations from newer (>0.5.3) MONAI versions.
    """

    def __init__(
        self,
        keys: KeysCollection,
        composed_transforms: Union[Collection[Compose], Compose],
        trace_key_postfix: str = "trace_dict",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.composed_transforms = ensure_tuple(composed_transforms)
        self.trace_key_postfix = trace_key_postfix

        assert len(self.keys) == len(self.composed_transforms), (
            f"The keys and composed_transforms must have the same length but got keys of length {len(self.keys)} and "
            f"composed_transforms of length {len(self.composed_transforms)}"
        )

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:

        d = dict(data)
        for k, ct in zip(self.keys, self.composed_transforms):
            d[f"{k}_binary_{self.trace_key_postfix}"] = self.__binary_trace(
                composed_transforms=ct
            )

            d[f"{k}_continuous_{self.trace_key_postfix}"] = self.__continuous_trace(
                composed_transforms=ct
            )
        return d

    @staticmethod
    def __binary_trace(composed_transforms):
        trace = OrderedDict()
        total_count = 0

        for transform in composed_transforms.transforms:

            trace[str(transform.__class__).split(".")[-1].split("'")[0]] = (
                transform._do_transform
                if isinstance(transform, RandomizableTransform)
                else True
            )

            total_count += 1

        return {"trace": trace, "total_count": total_count}

    @staticmethod
    def __continuous_trace(composed_transforms):
        trace = OrderedDict()
        total_count = 0
        for transform in composed_transforms.transforms:
            transform_key = str(transform.__class__).split(".")[-1].split("'")[0]

            if isinstance(transform, RandomizableTransform):
                if isinstance(transform, RandFlipd):
                    conditioning = [int(transform._do_transform)]
                elif isinstance(transform, RandRotate90d):
                    conditioning = [int(transform._do_transform) * transform._rand_k]
                elif isinstance(transform, RandAffined):
                    if (
                        transform._do_transform
                        and len(transform.rand_affine.rand_affine_grid.rotate_params)
                        > 0
                    ):
                        conditioning = (
                            transform.rand_affine.rand_affine_grid.rotate_params
                        )
                    else:
                        conditioning = [0, 0, 0]

                    if (
                        transform._do_transform
                        and len(transform.rand_affine.rand_affine_grid.shear_params) > 0
                    ):
                        conditioning += (
                            transform.rand_affine.rand_affine_grid.shear_params
                        )
                    else:
                        conditioning += [0, 0, 0]

                    if (
                        transform._do_transform
                        and len(transform.rand_affine.rand_affine_grid.translate_params)
                        > 0
                    ):
                        conditioning += (
                            transform.rand_affine.rand_affine_grid.translate_params
                        )
                    else:
                        conditioning += [0, 0, 0]

                    if (
                        transform._do_transform
                        and len(transform.rand_affine.rand_affine_grid.scale_params) > 0
                    ):
                        conditioning += (
                            transform.rand_affine.rand_affine_grid.scale_params
                        )
                    else:
                        conditioning += [1, 1, 1]

                elif isinstance(transform, RandAdjustContrastd):
                    conditioning = (
                        [transform.gamma_value] if transform._do_transform else [1]
                    )

                elif isinstance(transform, RandShiftIntensityd):
                    conditioning = [int(transform._do_transform) * transform._offset]
                elif isinstance(transform, RandGaussianNoised):
                    conditioning = [int(transform._do_transform)]
                else:
                    raise ValueError(
                        f"{transform_key} transformation is not supported."
                    )
            else:
                conditioning = [1]

            for idx, param in enumerate(conditioning):
                trace[f"{transform_key}_{idx}"] = param
                total_count += 1

        return {"trace": trace, "total_count": total_count}
