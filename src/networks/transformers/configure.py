import numpy as np

from src.networks.transformers.performer import Performer
from src.networks.transformers.transformer import TransformerBase
from src.networks.transformers.xtransformer import XTransformer
from src.utils.constants import (
    TransformerPositionalConditioningTypes,
    TransformerNetworks,
)


def get_transformer_network(config: dict) -> TransformerBase:

    if config["network"] == TransformerNetworks.PERFORMER.value:
        network = Performer(
            causal=True,
            ordering=config["ordering"],
            num_tokens=config["vocab_size"] + 1,
            max_seq_len=np.prod(config["spatial_shape"]) + 1,
            dim=config["n_embd"],
            depth=config["n_layers"],
            heads=config["n_head"],
            local_attn_heads=config["local_attn_heads"],
            local_window_size=config["local_window_size"],
            feature_redraw_interval=config["feature_redraw_interval"],
            generalized_attention=config["generalized_attention"],
            emb_dropout=config["emb_dropout"],
            ff_dropout=config["ff_dropout"],
            attn_dropout=config["attn_dropout"],
            use_scalenorm=config["use_scalenorm"],
            use_rezero=config["use_rezero"],
            tie_embed=config["tie_embedding"],
            rotary_position_emb=config["position_emb"]
            == TransformerPositionalConditioningTypes.ROTARY.value,
            fixed_position_emb=config["position_emb"]
            == TransformerPositionalConditioningTypes.FIXED.value,
            axial_position_emb=False,
            spatial_position_emb=config["spatial_position_emb"],
            spatial_shape=config["spatial_shape"],
            conditioning_num_tokens=config["vqvae_aug_conditioning_num_tokens"]
            + config["conditioning_num_tokens"],
            conditioning_type=config["conditioning_type"],
        )
    elif config["network"] == TransformerNetworks.XTRANSFORMER.value:
        network = XTransformer(
            dim=config["n_embd"],
            depth=config["n_layers"],
            num_tokens=config["vocab_size"] + 1,
            max_seq_len=np.prod(config["spatial_shape"]) + 1,
            ordering=config["ordering"],
            emb_dropout=config["emb_dropout"],
            ff_dropout=config["ff_dropout"],
            attn_dropout=config["attn_dropout"],
            tie_embedding=config["tie_embedding"],
            use_pos_emb=config["position_emb"]
            == TransformerPositionalConditioningTypes.FIXED.value,
            heads=config["n_head"],
            use_rezero=config["use_rezero"],
            rotary_pos_emb=config["position_emb"]
            == TransformerPositionalConditioningTypes.ROTARY.value,
            use_rmsnorm=config["use_rmsnorm"],
            ff_glu=config["ff_glu"],
            attn_talking_heads=config["attn_talking_heads"],
            attn_on_attn=config["attn_on_attn"],
            attn_gate_values=config["attn_gate_values"],
            sandwich_coef=config["sandwich_coef"],
            sandwich_norm=config["sandwich_norm"],
            rel_pos_bias=config["rel_pos_bias"],
            spatial_rel_pos_bias=config["spatial_rel_pos_bias"],
            relative_spatial_pos_attr=config["relative_spatial_pos_attr"],
            use_qk_norm_attn=config["use_qk_norm_attn"],
            spatial_position_emb=config["spatial_position_emb"],
            spatial_shape=config["spatial_shape"],
            conditioning_num_tokens=config["vqvae_aug_conditioning_num_tokens"]
            + config["conditioning_num_tokens"],
            conditioning_type=config["conditioning_type"],
            gate_residual=config["gate_residual"],
            shift_mem_down=config["shift_mem_down"],
            residual_attn=config["residual_attn"],
            cross_residual_attn=config["cross_residual_attn"],
            pre_norm=not (config["residual_attn"] or config["cross_residual_attn"]),
        )
    else:
        raise ValueError(
            f"Transformer unknown. Was given {config['network']} but choices are {[transformer.value for transformer in TransformerNetworks]}."
        )

    return network
