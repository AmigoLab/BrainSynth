from enum import Enum


class VQVAELosses(Enum):
    BASELINE = "baseline"
    BAUR = "baur"
    MSE = "mse"
    SPECTRAL = "spectral"
    HARTLEY = "hartley"
    JUKEBOX = "jukebox"
    WAVEGAN = "wavegan"
    PERCEPTUAL = "perceptual"
    JUKEBOX_PERCEPTUAL = "jukebox_perceptual"
    HARTLEY_PERCEPTUAL = "hartley_perceptual"
    JUKEBOX_PERCEPTUAL_SSIM = "jukebox_perceptual_ssim"
    SSIM = "ssim"
    SSIM_HARTLEY = "ssim_hartley"
    SSIM_MED3D = "ssim_med3d"
