from src.networks.feature_extractors.mednet3d import MedNet3D, BasicBlock, Bottleneck
from src.utils.constants import FeatureExtractorNetworks
import torch


def get_feature_extractor(config: dict) -> torch.nn.Module:
    if config["network"] == FeatureExtractorNetworks.MEDNET3D.value:
        if config["depth"] == 10:
            layers = [1, 1, 1, 1]
            shortcut_type = "B"
            block = BasicBlock
        elif config["depth"] == 18:
            layers = [2, 2, 2, 2]
            shortcut_type = "A"
            block = BasicBlock
        elif config["depth"] == 34:
            layers = [3, 4, 6, 3]
            shortcut_type = "A"
            block = BasicBlock
        elif config["depth"] == 50:
            layers = [3, 4, 6, 3]
            shortcut_type = "B"
            block = Bottleneck
        elif config["depth"] == 101:
            layers = [3, 4, 23, 3]
            shortcut_type = "B"
            block = Bottleneck
        elif config["depth"] == 152:
            layers = [3, 8, 36, 3]
            shortcut_type = "B"
            block = Bottleneck
        elif config["depth"] == 200:
            layers = [3, 24, 36, 3]
            shortcut_type = "B"
            block = Bottleneck
        else:
            raise ValueError(
                f"Received depth {config['depth']} for network {config['network']}, but available ones are "
                f"[10, 18, 34, 50, 101, 152, 200]"
            )

        network = MedNet3D(
            block=block, layers=layers, shortcut_type=shortcut_type, no_cuda=False
        )
    else:
        raise ValueError(
            f"Received network {config['network']} but the available ones are {[e.value for e in FeatureExtractorNetworks]}"
        )

    __load_model_weights(config=config, network=network)

    return network


def __load_model_weights(config: dict, network: torch.nn.Module) -> None:
    if config["network"] == FeatureExtractorNetworks.MEDNET3D.value:
        # This loading strategy was taken from MedNet3D codebase since their checkpoints did not include the segmentation
        #   heads so they need to be copied from the initialized network.
        #   https://github.com/Tencent/MedicalNet/blob/master/model.py#L87
        net_dict = network.state_dict()
        # Cleaning the name of the methods due to the fact that the weights were saved from a DDP/DP model instead of
        #   the underlying .module attribute
        pretrain_dict = torch.load(config["checkpoint_path"])["state_dict"]
        pretrain_dict = {k.replace("module.", ""): v for k, v in pretrain_dict.items()}

        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        network.load_state_dict(net_dict)
