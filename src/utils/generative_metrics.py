from collections import defaultdict
from typing import Union, Dict, List, Tuple, Sequence
from copy import deepcopy
from prdc import compute_prdc

import numpy as np
import torch
import xmltodict
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms import (
    LoadImaged,
    Lambdad,
    Compose,
    DivisiblePadd,
    NormalizeIntensityd,
    ThresholdIntensityd,
    AddChanneld,
    ScaleIntensityd,
    CenterSpatialCropd,
    SpatialCropd,
    SpatialPadd,
    SqueezeDimd,
    RepeatChanneld,
    Resized,
    ToTensord,
)
from monai.utils.enums import NumpyPadMode
from piq import FID
from piq.feature_extractors import InceptionV3
from scipy.stats import ttest_ind, wasserstein_distance
from torch import Tensor
from torch import device as torch_device

from src.networks.feature_extractors.configure import get_feature_extractor
from src.utils.ssim import SSIM
from tqdm import tqdm

RANDOM_SEED = 0


def softplus(x):
    """numerically stable calcuation for log(1 + exp(x))"""
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def softminus(x):
    return -softplus(-x)


def softclip(x, a=None, b=None, c=np.log(2) * (np.e * np.e + 1) * 0.5):
    """
    Clipping with softplus and softminus, with paramterized corner sharpness.
    Set either (or both) endpoint to None to indicate no clipping at that end.
    """
    # when clipping at both ends, make c dimensionless w.r.t. b - a / 2
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = x
    if a is not None:
        v = v - softminus(c * (x - a)) / c
    if b is not None:
        v = v - softplus(c * (x - b)) / c
    return v


@torch.no_grad()
def compute_3d_feats(
    loader: torch.utils.data.DataLoader,
    feature_extractor: torch.nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    r"""Generate low-dimensional image desciptors.

    It is a modified version of the original one:
        https://github.com/photosynthesis-team/piq/blob/244865e3e80ec22ae075e7dfef8ea3ce10a55891/piq/base.py#L18

    Args:
        loader: Should return dict with key `images` in it
        feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
            Model should return a list with features from one of the network layers.
        out_features: size of `feature_extractor` output
        device: Device on which to compute inference of the model
    """

    assert isinstance(
        feature_extractor, torch.nn.Module
    ), f"Feature extractor must be PyTorch module. Got {type(feature_extractor)}"
    feature_extractor.to(device)
    feature_extractor.eval()

    total_feats = []
    for batch in loader:
        images = batch["images"]
        images = images.float().to(device)

        # Get features
        features = feature_extractor(images)
        assert (
            len(features) == 1
        ), f"feature_encoder must return list with features from one layer. Got {len(features)}"
        total_feats.append(torch.mean(features[0], dim=(2, 3, 4)))

    return torch.cat(total_feats, dim=0)


def __mmd(synthetic_sample: Tensor, real_sample: Tensor) -> Tensor:
    synthetic_sample = synthetic_sample.view(synthetic_sample.shape[0], -1)
    real_sample = real_sample.view(real_sample.shape[0], -1)

    xx, yy, zz = (
        torch.mm(synthetic_sample, synthetic_sample.t()),
        torch.mm(real_sample, real_sample.t()),
        torch.mm(synthetic_sample, real_sample.t()),
    )
    xx = xx / real_sample.shape[1]
    yy = yy / real_sample.shape[1]
    zz = zz / real_sample.shape[1]

    # Beta = 1.0, Gamma = 2.0
    return 1.0 * (torch.mean(xx) + torch.mean(yy)) - 2.0 * torch.mean(zz)


def __to_numpy(x: Tensor, scalarize: bool = True) -> Union[float, int, complex]:
    x = x.detach().cpu().numpy()

    if scalarize:
        x = np.asscalar(x)

    return x


def __get_gif_dict(xmls_paths: List[str]) -> Dict[str, List[float]]:
    to_be_returned = defaultdict(lambda: list())
    missing_files = []
    for xml_path in xmls_paths:
        try:
            json_file = xmltodict.parse(open(xml_path).read())

            for tissue in json_file["document"]["tissues"]["item"]:
                to_be_returned[tissue["name"].replace(" ", "_").lower()].append(
                    float(tissue["volumeProb"])
                )

            for label in json_file["document"]["labels"]["item"]:
                to_be_returned[label["name"].replace(" ", "_").lower()].append(
                    float(label["volumeProb"])
                )
        except FileNotFoundError:
            missing_files.append(xml_path)

    if len(missing_files) != 0:
        print(
            f"Total files missing is {len(missing_files)} from {len(xmls_paths)}.\nThose are: {missing_files}"
        )

    return to_be_returned


def get_dataloaders(
    real_subjects: List[Dict[str, str]],
    synthetic_subjects: List[Dict[str, str]],
    spatial_dims: int,
    batch_size: int,
    pre_feature_extractor_normalization: bool,
    normalizes: Tuple[bool, bool],
    clamps: Tuple[bool, bool],
    soft_clamps: Tuple[bool, bool],
    divisible_pads_16: Tuple[bool, bool],
    rois: Tuple[
        Union[Tuple[int, ...], Tuple[Tuple[int, int], ...], None],
        Union[Tuple[int, ...], Tuple[Tuple[int, int], ...], None],
    ],
    roi_slice: Union[Sequence[slice], None] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    def get_transforms(
        normalize: bool,
        clamp: bool,
        soft_clamp: bool,
        roi: Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]],
        divisible_pad_16: bool,
        roi_slice_crop: Union[Sequence[slice], None],
        pfe_normalization: bool,
    ):
        transform = [
            LoadImaged(
                keys=["images", "ssim_images"],
                reader="NibabelReader",
                as_closest_canonical=True,
            ),
            AddChanneld(keys=["images", "ssim_images"]),
        ]

        if normalize:
            transform += [
                ScaleIntensityd(
                    keys=["ssim_images"] + (["images"] if pfe_normalization else []),
                    minv=0.0,
                    maxv=1.0,
                )
            ]

        if clamp:
            transform += [
                ThresholdIntensityd(
                    keys=["ssim_images"] + (["images"] if pfe_normalization else []),
                    threshold=1,
                    above=False,
                    cval=1.0,
                ),
                ThresholdIntensityd(
                    keys=["ssim_images"] + (["images"] if pfe_normalization else []),
                    threshold=0,
                    above=True,
                    cval=0,
                ),
            ]

        if soft_clamp:
            transform += [
                Lambdad(
                    keys=["ssim_images"] + (["images"] if pfe_normalization else []),
                    func=lambda x: softclip(x, a=0, b=1),
                )
            ]

        if roi:
            if type(roi[0]) is int:
                transform += [
                    CenterSpatialCropd(keys=["images", "ssim_images"], roi_size=roi)
                ]
            elif type(roi[0]) is tuple:
                transform += [
                    SpatialCropd(
                        keys=["images", "ssim_images"],
                        roi_start=[a[0] for a in roi],
                        roi_end=[a[1] for a in roi],
                    )
                ]
            else:
                raise ValueError(
                    f"roi should be either a Tuple with three ints like (0,1,2) or a Tuple with three Tuples that have "
                    f"two ints like ((0,1),(2,3),(4,5)). But received {roi}."
                )

            transform += [
                # This is here to guarantee no sample has lower spatial resolution than the ROI
                # YOU SHOULD NOT RELY ON THIS TO CATCH YOU SLACK, ALWAYS CHECK THE SPATIAL SIZES
                # OF YOU DATA PRIOR TO TRAINING ANY MODEL.
                SpatialPadd(
                    keys=["images", "ssim_images"],
                    spatial_size=roi
                    if type(roi[0]) is int
                    else [a[1] - a[0] for a in roi],
                    mode=NumpyPadMode.SYMMETRIC,
                )
            ]

        if divisible_pad_16:
            transform += [
                DivisiblePadd(
                    keys=["images", "ssim_images"], k=16, mode=NumpyPadMode.EDGE
                )
            ]

        if roi_slice_crop:
            transform += [
                SpatialCropd(keys=["images", "ssim_images"], roi_slices=roi_slice_crop),
                SqueezeDimd(keys=["images", "ssim_images"], dim=None),
                AddChanneld(keys=["images", "ssim_images"]),
                RepeatChanneld(keys=["images", "ssim_images"], repeats=3),
            ]

        if spatial_dims == 3:
            transform += [NormalizeIntensityd(keys=["images"], channel_wise=True)]

        transform += [ToTensord(keys=["images", "ssim_images"])]

        return Compose(transform)

    real_dataloader = DataLoader(
        Dataset(
            data=real_subjects,
            transform=get_transforms(
                soft_clamp=soft_clamps[0],
                clamp=clamps[0],
                normalize=normalizes[0],
                roi=rois[0],
                divisible_pad_16=divisible_pads_16[0],
                roi_slice_crop=roi_slice,
                pfe_normalization=pre_feature_extractor_normalization,
            ),
        ),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        pin_memory=False,
        prefetch_factor=6,
        drop_last=True,
    )

    synthetic_dataloader = DataLoader(
        Dataset(
            data=synthetic_subjects,
            transform=get_transforms(
                soft_clamp=soft_clamps[1],
                clamp=clamps[1],
                normalize=normalizes[1],
                roi=rois[1],
                divisible_pad_16=divisible_pads_16[1],
                roi_slice_crop=roi_slice,
                pfe_normalization=pre_feature_extractor_normalization,
            ),
        ),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        pin_memory=False,
        prefetch_factor=6,
        drop_last=True,
    )

    return real_dataloader, synthetic_dataloader


def calculate_metrics(
    spatial_dims: int,
    experiment: str,
    output_folder: str,
    real_dataloader: DataLoader,
    synthetic_dataloader: DataLoader,
    med3d_config: dict,
    device: torch_device = torch_device("cuda"),
    batched_mmd: int = -1,
    calculate_reconstruction_metrics: bool = True,
    nearest_k: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    ms_ssim = SSIM(
        spatial_dims=spatial_dims,
        in_channels=3 if spatial_dims == 2 else 1,
        data_range=1.0,
        size_average=False,
        gaussian_kernel_size=7,
        gaussian_kernel_sigma=1.5,
        k=(0.01, 0.03),
        scales=(1, 1, 1),
        multi_scale_weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        gradient_based=False,
        gradient_masks_weights=None,
        star_based=False,
    ).to(device)

    ssim = SSIM(
        spatial_dims=spatial_dims,
        in_channels=3 if spatial_dims == 2 else 1,
        data_range=1.0,
        size_average=False,
        gaussian_kernel_size=7,
        gaussian_kernel_sigma=1.5,
        k=(0.01, 0.03),
        scales=(1, 1, 1),
        multi_scale_weights=None,
        gradient_based=False,
        gradient_masks_weights=None,
        star_based=False,
    ).to(device)

    ms_ssim_4_g = SSIM(
        spatial_dims=spatial_dims,
        in_channels=3 if spatial_dims == 2 else 1,
        data_range=1.0,
        size_average=False,
        gaussian_kernel_size=7,
        gaussian_kernel_sigma=1.5,
        k=(0.01, 0.03),
        scales=(1, 1, 1),
        multi_scale_weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        gradient_based=True,
        gradient_masks_weights=(0.25, 0.25, 0.25, 0.25),
        star_based=False,
    ).to(device)

    ssim_4_g = SSIM(
        spatial_dims=spatial_dims,
        in_channels=3 if spatial_dims == 2 else 1,
        data_range=1.0,
        size_average=False,
        gaussian_kernel_size=7,
        gaussian_kernel_sigma=1.5,
        k=(0.01, 0.03),
        scales=(1, 1, 1),
        multi_scale_weights=None,
        gradient_based=True,
        gradient_masks_weights=(0.25, 0.25, 0.25, 0.25),
        star_based=False,
    ).to(device)

    to_be_returned = {"ms_ssim": [], "ssim": [], "4_g_ms_ssim": [], "4_g_ssim": []}

    feature_extractor = (
        get_feature_extractor(med3d_config)
        if spatial_dims == 3
        else InceptionV3(requires_grad=False)
    )
    feature_extractor = feature_extractor.to(device=device)
    feature_extractor.eval()

    if batched_mmd > 0:
        to_be_returned["mmd"] = []
        with torch.no_grad():
            for real_samples, synthetic_samples in zip(
                real_dataloader, synthetic_dataloader
            ):
                real_samples = feature_extractor(real_samples["images"].to(device))[0]
                synthetic_samples = feature_extractor(
                    synthetic_samples["images"].to(device)
                )[0]

                to_be_returned["mmd"].append(
                    __to_numpy(__mmd(real_samples, synthetic_samples))
                )

    for real_samples, synthetic_samples in tqdm(
        zip(
            real_dataloader
            if calculate_reconstruction_metrics
            else deepcopy(synthetic_dataloader),
            synthetic_dataloader,
        )
    ):
        real_samples = real_samples["ssim_images"].to(device)
        synthetic_samples = synthetic_samples["ssim_images"].to(device)

        to_be_returned["ms_ssim"].append(
            __to_numpy(ms_ssim(real_samples, synthetic_samples), scalarize=False)
        )

        to_be_returned["ssim"].append(
            __to_numpy(ssim(real_samples, synthetic_samples), scalarize=False)
        )

        to_be_returned["4_g_ms_ssim"].append(
            __to_numpy(ms_ssim_4_g(real_samples, synthetic_samples), scalarize=False)
        )

        to_be_returned["4_g_ssim"].append(
            __to_numpy(ssim_4_g(real_samples, synthetic_samples), scalarize=False)
        )

        del real_samples, synthetic_samples

    raw_values = {}

    for key in list(to_be_returned.keys()):
        values = to_be_returned.pop(key)
        to_be_returned[f"{key}_mean"] = np.mean(values)
        to_be_returned[f"{key}_std"] = np.std(values)
        raw_values[key] = values

    fid_metric = FID()

    real_features = fid_metric.compute_feats(
        loader=real_dataloader, feature_extractor=feature_extractor, device=device
    )

    synthetic_features = fid_metric.compute_feats(
        loader=synthetic_dataloader, feature_extractor=feature_extractor, device=device
    )

    to_be_returned["fid"] = __to_numpy(fid_metric(real_features, synthetic_features))

    if batched_mmd == -1:
        to_be_returned["mmd_mean"] = __to_numpy(
            __mmd(real_features, synthetic_features)
        )
        to_be_returned["mmd_std"] = 0

    real_features = real_features.cpu().numpy()
    synthetic_features = synthetic_features.cpu().numpy()

    metrics = compute_prdc(
        real_features=real_features,
        fake_features=synthetic_features,
        nearest_k=nearest_k,
    )

    to_be_returned["precision"] = metrics["precision"]
    to_be_returned["recall"] = metrics["recall"]
    to_be_returned["density"] = metrics["density"]
    to_be_returned["coverage"] = metrics["coverage"]

    np.save(file=f"{output_folder}_{experiment}_real_features.npy", arr=real_features)
    np.save(
        file=f"{output_folder}synthetic_{experiment}_features.npy",
        arr=synthetic_features,
    )

    return to_be_returned, raw_values


def calculate_gif_metrics(
    real_gifs: Dict[str, List[str]],
    synthetic_gifs: Dict[str, List[str]],
    target_p_value=0.001,
    bonferroni_correction=True,
    bonferroni_tissue_count=None,
    decimal_precision=2,
):
    to_be_returned = defaultdict(lambda: list())

    for key in list(real_gifs.keys()):
        real_gifs[key] = __get_gif_dict(xmls_paths=real_gifs[key])
        synthetic_gifs[key] = __get_gif_dict(xmls_paths=synthetic_gifs[key])

    if bonferroni_correction:
        # 216 is the total number of labels (208) and tissues (8) that GIF outputs in the XML file
        target_p_value = target_p_value / (
            len(real_gifs)
            * (216 if bonferroni_tissue_count is None else bonferroni_tissue_count)
        )

    for experiment in synthetic_gifs.keys():
        synthetic_gif = synthetic_gifs[experiment]
        real_gif = real_gifs[experiment]
        to_be_returned["experiment"].append(experiment)
        for tissue in real_gif.keys():
            statistic, pvalue = ttest_ind(real_gif[tissue], synthetic_gif[tissue])
            wd = wasserstein_distance(
                u_values=real_gif[tissue], v_values=synthetic_gif[tissue]
            )
            mean_synth_tissue = np.mean(synthetic_gif[tissue])
            std_synth_tissue = np.std(synthetic_gif[tissue])
            mean_real_tissue = np.mean(real_gif[tissue])
            std_real_tissue = np.std(real_gif[tissue])

            statistic = np.round(statistic, decimals=decimal_precision)
            wd = np.round(np.log(wd), decimals=decimal_precision)
            mean_synth_tissue = np.round(mean_synth_tissue, decimals=-3) / 1000
            std_synth_tissue = np.round(std_synth_tissue, decimals=-3) / 1000
            mean_real_tissue = np.round(mean_real_tissue, decimals=-3) / 1000
            std_real_tissue = np.round(std_real_tissue, decimals=-3) / 1000

            to_be_returned[f"synthetic_mean_{tissue}"].append(mean_synth_tissue)
            to_be_returned[f"synthetic_std_{tissue}"].append(std_synth_tissue)
            to_be_returned[f"real_mean_{tissue}"].append(mean_real_tissue)
            to_be_returned[f"real_std_{tissue}"].append(std_real_tissue)
            to_be_returned[f"t_value_{tissue}"].append(statistic)
            to_be_returned[f"p_value_{tissue}"].append(pvalue)
            to_be_returned[f"significant_{tissue}"].append(pvalue < target_p_value)
            to_be_returned[f"wasserstein_{tissue}"].append(wd)

    return to_be_returned, target_p_value
