from typing import TypedDict, List, Tuple, Dict

import torch
from torch.nn import Sequential, Conv1d, AvgPool1d, ReLU, Flatten, Linear
import pyro.distributions.transforms as trans

from mdshvb.transforms import SoftclipTransform, BatchedStackTransform


class HRFAnglePriorConfig(TypedDict):
    uniform_low: float
    uniform_high: float


class LatentScalePriorConfig(TypedDict):
    normal_loc: float  # > 0
    normal_scale: float  # > 0


class BOLDScalePriorConfig(TypedDict):
    normal_loc: float  # > 0
    normal_scale: float  # > 0


class NFConfig(TypedDict):
    n_blocks: int
    hidden_dims: List[int]


class EncoderConfig(TypedDict):
    conv1d_channel_kernel_strides: List[Tuple[int, int, int]]
    avgpool1d_kernels: List[int]
    flat_size: int


class NFEncoderConfig(TypedDict):
    nf_config: NFConfig
    encoder_config: EncoderConfig
    context_dim: int


def get_conditional_density_estimator(
    context_dim: int, nf_config: NFConfig
) -> trans.ComposeTransformModule:
    """Get normalizing flow that approximates the HP posterior
    conditional to the signal encoding

    Args:
        context_dim (int)
        nf_config (NFConfig): from hydra

    Returns:
        trans.ComposeTransformModule: flow
    """
    transforms = []
    for _ in range(nf_config["n_blocks"]):
        transform = trans.conditional_affine_autoregressive(
            input_dim=3,
            context_dim=context_dim,
            hidden_dims=nf_config["hidden_dims"],
            log_scale_min_clip=-5.0,
            log_scale_max_clip=3.0,
            sigmoid_bias=2.0,
            stable=True,
        )
        transforms.append(transform)

    nf = trans.ComposeTransformModule(parts=transforms)

    return nf


def get_encoder(context_dim: int, encoder_config: EncoderConfig) -> Sequential:
    """Get encoder for the observed BOLD signal

    Args:
        context_dim (int)
        encoder_config (EncoderConfig): from hydra

    Returns:
        Sequential: conv1D encoder
    """
    layers = []
    prev_channels = 1

    for (out_channels, conv_kernel_size, stride), avg_kernel_size in zip(
        encoder_config["conv1d_channel_kernel_strides"],
        encoder_config["avgpool1d_kernels"],
    ):
        layers.append(
            Conv1d(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=stride,
            )
        )
        layers.append(ReLU())
        layers.append(AvgPool1d(kernel_size=avg_kernel_size))

        prev_channels = out_channels

    layers.append(Flatten(start_dim=-2, end_dim=-1))
    layers.append(
        Linear(
            in_features=encoder_config["flat_size"], out_features=context_dim
        )
    )
    layers.append(ReLU())

    encoder = Sequential(*layers)

    return encoder


def get_nf_encoder(
    nf_encoder_config: NFEncoderConfig,
) -> Tuple[trans.ComposeTransformModule, Sequential]:
    """get (flow, encoder) couple that constitutes the HP estimator

    Args:
        nf_encoder_config (NFEncoderConfig): from hydra

    Returns:
        Tuple[trans.ComposeTransformModule, Sequential]: nf, encoder
    """
    context_dim = nf_encoder_config["context_dim"]

    nf = get_conditional_density_estimator(
        context_dim, nf_encoder_config["nf_config"]
    )
    encoder = get_encoder(context_dim, nf_encoder_config["encoder_config"])

    return nf, encoder


def unstack_alpha_r_q(
    stacked_alpha_r_q: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    alpha = stacked_alpha_r_q[..., 0]
    r = stacked_alpha_r_q[..., 1]
    q = stacked_alpha_r_q[..., 2]

    return dict(alpha=alpha, r=r, q=q)


def stack_alpha_r_q(
    alpha: torch.Tensor, r: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    stacked_alpha_r_q = torch.stack([alpha, r, q], dim=-1)

    return stacked_alpha_r_q


def get_transform(
    hrf_angle_prior_config: HRFAnglePriorConfig,
    bold_scale_prior_config: BOLDScalePriorConfig,
    latent_scale_prior_config: LatentScalePriorConfig,
) -> trans.StackTransform:
    """Transform that combines:
    - soft clipping for alpha (HRF angle)
    - softplus for r and q (variance levels)
    to apply on top of flow

    Args:
        hrf_angle_prior_config (HRFAnglePriorConfig): from hydra
        bold_scale_prior_config (BOLDScalePriorConfig): from hydra
        latent_scale_prior_config (LatentScalePriorConfig): from hydra

    Returns:
        trans.StackTransform: to apply on top of flow
    """
    return BatchedStackTransform(
        [
            SoftclipTransform(
                low=hrf_angle_prior_config["uniform_low"],
                high=hrf_angle_prior_config["uniform_high"],
                slope=2.0,
            ),
            trans.ComposeTransform(
                [
                    trans.AffineTransform(
                        loc=bold_scale_prior_config["normal_loc"],
                        scale=bold_scale_prior_config["normal_scale"],
                    ),
                    trans.SoftplusTransform(),
                ]
            ),
            trans.ComposeTransform(
                [
                    trans.AffineTransform(
                        loc=latent_scale_prior_config["normal_loc"],
                        scale=latent_scale_prior_config["normal_scale"],
                    ),
                    trans.SoftplusTransform(),
                ]
            ),
        ],
        dim=-1,
        reinterpreted_batch_ndims=1,
    )
