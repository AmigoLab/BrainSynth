from typing import List, Any

import copy
import torch
from monai.inferers import Inferer
import numpy as np
from torch.nn import functional as F
from pathlib import Path

from src.networks.vqvae.baseline import BaselineVQVAE
from src.networks.transformers.transformer import TransformerBase

class LikelihoodMapInferer(Inferer):
    def __init__(self, transf_network: TransformerBase, device: Any, output_dir: str) -> None:
        Inferer.__init__(self)

        self.transformer_network = transf_network
        self.device = device

    def __call__(self, inputs: torch.Tensor,  network: BaselineVQVAE, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        embedding_indices = network.index_quantize(images=inputs)
        recons = network.decode_samples(embedding_indices=embedding_indices)
        embedding_indices = embedding_indices[0]
        latent_shape = embedding_indices.shape
        index_sequence = self.transformer_network.ordering.get_sequence_ordering()
        revert_ordering = self.transformer_network.ordering.get_revert_sequence_ordering()

        embedding_indices = embedding_indices.reshape(embedding_indices.shape[0], -1)
        embedding_indices = embedding_indices[:, index_sequence]


        likelihood_map= self.get_likelihood_maps(embedding_indices, latent_shape, revert_ordering)

        embedding_indices = embedding_indices.reshape(latent_shape)
        embedding_indices = embedding_indices.cpu().numpy()

        outputs = {"likelihood_map": likelihood_map,
                   'input':inputs,
                   'recon':recons,
                   'recon_error':recons-inputs,}

        return outputs

    @torch.no_grad()
    def get_likelihood_maps(self, zs, latent_shape, rev_ordering):

        vocab_size = self.transformer_network.vocab_size-1
        zs = F.pad(zs, (1, 0), "constant", vocab_size)

        zs_in = zs[:, :-1]
        zs_out = zs[:, 1:]

        logits = self.transformer_network(zs_in)
        probs = F.softmax(logits, dim=-1).cpu()
        selected_probs = torch.gather(probs, 2, zs_out.cpu().unsqueeze(2).long())
        selected_probs = selected_probs.squeeze(2)

        mask = copy.deepcopy(selected_probs)
        mask = mask[:, rev_ordering]
        mask= mask.reshape(latent_shape)
        mask = mask.cpu().numpy()

        return mask

