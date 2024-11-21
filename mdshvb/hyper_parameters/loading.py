from typing import Tuple
import torch
import pyro
from mdshvb.hyper_parameters.alpha_r_q import NFEncoderConfig, get_nf_encoder


def load_nf_encoder(
    nf_weights_filepath: str,
    encoder_weights_filepath: str,
    nf_encoder_config: NFEncoderConfig,
) -> Tuple[
    pyro.distributions.transforms.ComposeTransformModule, torch.nn.Sequential
]:
    """Load trained HP estimator

    Args:
        nf_weights_filepath (str)
        encoder_weights_filepath (str)
        nf_encoder_config (NFEncoderConfig): from hydra

    Returns:
        Tuple[ pyro.distributions.transforms.ComposeTransformModule, torch.nn.Sequential ]: nf, encoder
    """

    nf, encoder = get_nf_encoder(
        nf_encoder_config=nf_encoder_config,
    )

    nf_state_dict = torch.load(nf_weights_filepath)
    nf.load_state_dict(nf_state_dict)
    nf.eval()

    encoder_state_dict = torch.load(encoder_weights_filepath)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    return nf, encoder
