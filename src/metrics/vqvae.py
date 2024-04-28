from typing import Sequence

import pytorch_msssim
import torch
import torch.nn.functional as F
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class MultiScaleSSIM(Metric):
    input_ranges = ["sigmoid", "tanh"]
    normalization_strategies = [None, "scaling", "clipping"]

    def __init__(
        self,
        output_transform=lambda x: x,
        ms_ssim_kwargs=None,
        normalization_strategy: str = "scaling",
        input_range: str = "sigmoid",
    ):
        self._accumulator = None
        self._count = None

        self._normalization_strategy = normalization_strategy.lower()
        assert (
            self._normalization_strategy in self.normalization_strategies,
            f"Got {self._normalization_strategy} but valid normalization strategies are {self.normalization_strategies}.",
        )

        self._input_range = input_range.lower()
        assert (
            self._input_range in self.input_ranges,
            f"Got {self._input_range} but valid input ranges are {self.input_ranges}.",
        )

        self._ms_ssim_kwargs = {
            "data_range": 1,
            "win_size": 11,
            "win_sigma": 1.5,
            "size_average": False,
            "weights": None,
            "K": (0.01, 0.03),
        }

        if ms_ssim_kwargs:
            self._ms_ssim_kwargs.update(ms_ssim_kwargs)

        super(MultiScaleSSIM, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MultiScaleSSIM, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        y = self.normalize_input(y)
        y_pred = self.normalize_input(y_pred)

        self._accumulator += torch.sum(
            pytorch_msssim.ms_ssim(X=y, Y=y_pred, **self._ms_ssim_kwargs)
        ).item()

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MultiScaleSSIM must have at least one example before it can be computed."
            )
        return self._accumulator / self._count

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid input range - [0,1]
        if self._input_range == self.input_ranges[1]:
            x = x + 1
            x = x / 2

        # Min-Max Scaling Normalization
        if self._normalization_strategy == self.normalization_strategies[0]:
            x = x - torch.amin(x, dim=(2, 3, 4), keepdim=True)
            x = x / torch.amax(x, dim=(2, 3, 4), keepdim=True)
        # Clipping Normalization
        elif self._normalization_strategy == self.normalization_strategies[1]:
            x = torch.clamp(x, min=0, max=1)

        return x


class MAE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MAE, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MAE, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        self._accumulator += (
            F.l1_loss(input=y_pred, target=y, reduction="mean")
        ).item() * y.shape[0]

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MAE must have at least one example before it can be computed."
            )
        return self._accumulator / self._count


class MSE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MSE, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MSE, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        self._accumulator += (
            F.mse_loss(input=y_pred, target=y, reduction="mean")
        ).item() * y.shape[0]

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MSE must have at least one example before it can be computed."
            )
        return self._accumulator / self._count
