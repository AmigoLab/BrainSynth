from enum import Enum


class AugmentationStrengthScalers(Enum):
    # This is needed since some augmentation might need sub/super-linear scaling
    AFFINEROTATE = 0.2
    AFFINETRANSLATE = 1
    AFFINESCALE = 0.01
    ADJUSTCONTRASTGAMMA = 0.01
    SHIFTINTENSITYOFFSET = 0.025
    GAUSSIANNOISESTD = 0.01


class VQVAENetworks(Enum):
    SINGLE_VQVAE = "single_vqvae"
    BASELINE_VQVAE = "baseline_vqvae"
    SLIM_VQVAE = "slim_vqvae"


class VQVAEModes(Enum):
    TRAINING = "training"
    EXTRACTING = "extracting"
    DECODING = "decoding"


class DecayWarmups(Enum):
    STEP = "step"
    LINEAR = "linear"
    NONE = "none"


class TransformerNetworks(Enum):
    PERFORMER = "performer"
    XTRANSFORMER = "xtransformer"


class TransformerModes(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class TransformerConditioningType(Enum):
    NONE = "none"
    BOSREPLACEMENT = "bos_replacement"
    PREPENDING = "prepending"
    CROSSATTEND = "cross_attend"


class TransformerSpatialConditioningType(Enum):
    FIXED = "fixed"
    ABSOLUTE = "absolute"


class TransformerPositionalConditioningTypes(Enum):
    FIXED = "fixed"
    ABSOLUTE = "absolute"
    ROTARY = "rotary"


class TransformerVQVAEConditioningTypes(Enum):
    NONE = "none"
    BINARY = "binary"
    CONTINUOUS = "continuous"

class FeatureExtractorNetworks(Enum):
    MEDNET3D = "mednet3d"
