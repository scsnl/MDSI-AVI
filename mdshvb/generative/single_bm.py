from typing import TypedDict, Literal, Dict, Callable
from functools import partial

import torch
import pyro

from torch import Size
import pyro.distributions as dist
import pyro.distributions.transforms as trans
from pyro.ops.indexing import Vindex

from mdshvb.hyper_parameters.alpha_r_q import (
    HRFAnglePriorConfig,
    LatentScalePriorConfig,
    BOLDScalePriorConfig,
    stack_alpha_r_q,
)
from mdshvb.generative.hrf import HRFConfig, get_spm_derivatives_hrf
from mdshvb.distributions import LinearGaussianMM


class CouplingMatrixPriorConfig(TypedDict):
    type: Literal["Laplace", "Uniform"]
    laplace_self_coupling_loc: float | None
    laplace_self_coupling_scale: float | None
    laplace_coupling_scale: float | None


class SingleModelConfig(TypedDict):
    n_regions: int
    n_timesteps: int
    n_conditions: int
    coupling_matrix_prior_config: CouplingMatrixPriorConfig
    latent_scale_prior_config: LatentScalePriorConfig
    bold_scale_prior_config: BOLDScalePriorConfig
    hrf_angle_prior_config: HRFAnglePriorConfig
    hrf_config: HRFConfig


class FKLDatasetConfig(TypedDict):
    single_model_config: SingleModelConfig
    dataset_size: int
    Y_max_abs_val: float
    seed: int


class AlphaRQPrior(dist.TorchDistribution):
    """
    Combination of:
    - uniform prior for alpha (HRF angle)
    - log-normal prior for r (BOLD-level noise variance)
    - log-normal prior for q (BOLD-level noise variance)
    """

    def __init__(
        self,
        n_regions: int,
        hrf_angle_prior_config: HRFAnglePriorConfig,
        bold_scale_prior_config: BOLDScalePriorConfig,
        latent_scale_prior_config: LatentScalePriorConfig,
        batch_size: Size = Size([]),
        validate_args: bool | None = False,
        device: str = "cpu",
    ):
        M = n_regions

        super().__init__(
            batch_shape=batch_size,
            event_shape=Size([M, 3]),
            validate_args=validate_args,
        )

        self.alpha_dist = (
            dist.Uniform(
                low=torch.tensor(hrf_angle_prior_config["uniform_low"]).to(
                    device
                ),
                high=torch.tensor(hrf_angle_prior_config["uniform_high"]).to(
                    device
                ),
            )
            .expand((M,))
            .to_event(1)
        )
        self.r_dist = (
            dist.TransformedDistribution(
                dist.Normal(
                    loc=torch.tensor(bold_scale_prior_config["normal_loc"]).to(
                        device
                    ),
                    scale=torch.tensor(
                        bold_scale_prior_config["normal_scale"]
                    ).to(device),
                ),
                trans.SoftplusTransform(),
            )
            .expand((M,))
            .to_event(1)
        )
        self.q_dist = (
            dist.TransformedDistribution(
                dist.Normal(
                    loc=torch.tensor(
                        latent_scale_prior_config["normal_loc"]
                    ).to(device),
                    scale=torch.tensor(
                        latent_scale_prior_config["normal_scale"]
                    ).to(device),
                ),
                trans.SoftplusTransform(),
            )
            .expand((M,))
            .to_event(1)
        )

    def sample(self, sample_shape: Size = Size([])) -> torch.Tensor:
        alpha = self.alpha_dist.sample(sample_shape)
        r = self.r_dist.sample(sample_shape)
        q = self.q_dist.sample(sample_shape)

        alpha_r_q = torch.stack([alpha, r, q], dim=-1)

        return alpha_r_q

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        alpha = value[..., 0]
        log_prob_alpha = self.alpha_dist.log_prob(alpha)

        r = value[..., 1]
        log_prob_r = self.r_dist.log_prob(r)

        q = value[..., 2]
        log_prob_q = self.q_dist.log_prob(q)

        log_prob = log_prob_alpha + log_prob_r + log_prob_q

        return log_prob


def get_single_model(
    single_model_config: SingleModelConfig, device: str = "cpu"
) -> Callable:
    """Creates a Pyro callable, from which it is possible to sample
    synthetic latent parameters and observed signal.

    Args:
        single_model_config (SingleModelConfig): from hydra config
        device (str, optional): Defaults to "cpu".

    Returns:
        Callable: generative model
    """
    M = single_model_config["n_regions"]
    T = single_model_config["n_timesteps"]
    C = single_model_config["n_conditions"]

    hrf_dict = get_spm_derivatives_hrf(
        n_regions=M,
        time_repetition=single_model_config["hrf_config"]["time_repetition"],
        time_duration_kernel=single_model_config["hrf_config"][
            "time_duration_kernel"
        ],
        device=device,
    )
    K = hrf_dict["n_kernel_timesteps"]

    def model(
        observed_Y: torch.Tensor | None = None,
        observed_c: torch.Tensor | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Hyper-parameters:
        # - HRF angle alpha
        # - BOLD-level scale r
        # - latent activation-level scale q
        alpha_r_q = pyro.sample(
            "alpha_r_q",
            AlphaRQPrior(
                n_regions=M,
                hrf_angle_prior_config=single_model_config[
                    "hrf_angle_prior_config"
                ],
                bold_scale_prior_config=single_model_config[
                    "bold_scale_prior_config"
                ],
                latent_scale_prior_config=single_model_config[
                    "latent_scale_prior_config"
                ],
                device=device,
            ),
        )

        alpha = pyro.deterministic("alpha", alpha_r_q[..., 0], event_dim=1)
        r = pyro.deterministic("r", alpha_r_q[..., 1], event_dim=1)
        q = pyro.deterministic("q", alpha_r_q[..., 2], event_dim=1)

        # coupling matrix A
        if (
            single_model_config["coupling_matrix_prior_config"]["type"]
            == "Uniform"
        ):
            A = pyro.sample(
                "A",
                dist.Uniform(
                    low=torch.tensor(-1).to(device),
                    high=torch.tensor(1).to(device),
                )
                .expand((C, M, M))
                .to_event(3),
            )
        elif (
            single_model_config["coupling_matrix_prior_config"]["type"]
            == "Laplace"
        ):
            A = pyro.sample(
                "A",
                dist.Laplace(
                    loc=single_model_config["coupling_matrix_prior_config"][
                        "laplace_self_coupling_loc"
                    ]
                    * torch.eye(M, M).to(device),
                    scale=single_model_config["coupling_matrix_prior_config"][
                        "laplace_self_coupling_scale"
                    ]
                    * torch.eye(M, M).to(device)
                    + single_model_config["coupling_matrix_prior_config"][
                        "laplace_coupling_scale"
                    ]
                    * (1 - torch.eye(M, M)).to(device),
                )
                .to_event(2)
                .expand((C,))
                .to_event(1),
            )

        # experimental condition c
        c = pyro.sample(
            "c",
            dist.Categorical(logits=torch.zeros(C).to(device))
            .expand((T,))
            .to_event(1),
            obs=observed_c,
        )
        A_at_each_timestep = Vindex(A[..., None, :, :, :])[..., c, :, :]

        # ? latent activation X

        x_noise_dit = dist.Normal(
            loc=torch.zeros(1, M).to(device),
            scale=q[..., None, :],
        ).to_event(1)

        X_transposed = pyro.sample(
            "X_transposed",
            LinearGaussianMM(
                initial_dist=dist.Normal(
                    loc=torch.zeros(M).to(device), scale=q
                ).to_event(1),
                transition_matrix=A_at_each_timestep,
                transition_dist=x_noise_dit,
                duration=T,
            ),
        )

        X = pyro.deterministic(
            "X", torch.transpose(X_transposed, -1, -2), event_dim=2
        )

        convolution_kernel = (
            torch.cos(alpha)[..., None, None] * hrf_dict["kernel"]
            + torch.cos(alpha)[..., None, None]
            * hrf_dict["kernel_time_derivative"]
        )

        # BOLD signal Y
        X_zero_padded = torch.cat(
            [torch.zeros(X.shape[:-2] + (M, K - 1)).to(device), X], dim=-1
        )[..., None, :, :]

        if len(X_zero_padded.shape) == 3:
            y_loc = torch.nn.functional.conv1d(
                X_zero_padded,
                convolution_kernel,
            )[..., 0, :, :]
        else:
            y_loc = torch.vmap(
                partial(torch.nn.functional.conv1d, padding="valid"),
                in_dims=-4,
                out_dims=-4,
            )(X_zero_padded, convolution_kernel)[..., 0, :, :]

        Y = pyro.sample(
            "Y",
            dist.Normal(loc=y_loc, scale=r[..., None]).to_event(2),
            obs=observed_Y,
        )

        return {"A": A, "c": c, "q": q, "X": X, "r": r, "alpha": alpha, "Y": Y}

    return model


def generate_fkl_dataset(
    dataset_config: FKLDatasetConfig,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic dataset for HP estimator training.

    Args:
        dataset_config (FKLDatasetConfig): from hydra

    Returns:
        Dict[str, torch.Tensor]: dict(
            Y=BOLD signal,
            alpha_r_q=hyper_parameters,
        )
    """

    pyro.set_rng_seed(dataset_config["seed"])

    n_samples = (
        dataset_config["dataset_size"]
        // dataset_config["single_model_config"]["n_regions"]
    )

    model = get_single_model(dataset_config["single_model_config"])

    with pyro.plate("samples", n_samples, dim=-1):
        sample = model()

    Y = sample["Y"]
    indicator = torch.logical_and(
        ~torch.isnan(Y).any(-1).any(-1),
        ~(torch.abs(Y) > dataset_config["Y_max_abs_val"]).any(-1).any(-1),
    )

    fkl_dataset = dict(
        Y=Y[indicator][..., None, :].reshape(
            (-1, 1, dataset_config["single_model_config"]["n_timesteps"])
        ),
        alpha_r_q=stack_alpha_r_q(
            sample["alpha"][indicator],
            sample["r"][indicator],
            sample["q"][indicator],
        ).reshape((-1, 3)),
    )

    return fkl_dataset
