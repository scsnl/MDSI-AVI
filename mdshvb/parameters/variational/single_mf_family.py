from collections import defaultdict
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints

from typing import Callable, Dict, TypedDict

import torch
from mdshvb.hyper_parameters.alpha_r_q import get_transform

from mdshvb.generative.single_bm import SingleModelConfig


class SingleMFGuideHyperparametersConfig(TypedDict):
    pass  # no parameters


class SingleMFGuideConfig(TypedDict):
    single_model_config: SingleModelConfig
    hyper_parameters: SingleMFGuideHyperparametersConfig


def get_single_mf_guide(
    single_mf_guide_config: SingleMFGuideConfig,
    hp_estimator_nf: dist.transforms.ComposeTransformModule,
    initial_values: Dict[str, torch.Tensor] = defaultdict(lambda: None),  # type: ignore
    device: str = "cpu",
) -> Callable:
    """Creates Pyro callable, from which it is possible to sample
    latent parameters

    Args:
        single_mf_guide_config (SingleMFGuideConfig): from hydra
        hp_estimator_nf (dist.transforms.ComposeTransformModule)
        initial_values (Dict[str, torch.Tensor], optional): Defaults to defaultdict(lambda: None).

    Returns:
        Callable: variational guide
    """

    M = single_mf_guide_config["single_model_config"]["n_regions"]
    C = single_mf_guide_config["single_model_config"]["n_conditions"]
    T = single_mf_guide_config["single_model_config"]["n_timesteps"]

    hp_estimator_nf.to(device)

    single_model_config = single_mf_guide_config["single_model_config"]
    alpha_r_q_transform = get_transform(
        hrf_angle_prior_config=single_model_config["hrf_angle_prior_config"],
        bold_scale_prior_config=single_model_config["bold_scale_prior_config"],
        latent_scale_prior_config=single_model_config[
            "latent_scale_prior_config"
        ],
    )

    def guide(observed_encodings: torch.Tensor, **kwargs):
        # Hyper-parameters:
        # - HRF angle alpha
        # - BOLD-level scale r
        # - latent activation-level scale q
        alpha_r_q = pyro.sample(
            "alpha_r_q",
            dist.TransformedDistribution(
                dist.ConditionalTransformedDistribution(
                    base_dist=dist.Normal(
                        loc=torch.zeros(M, 3).to(device),
                        scale=torch.ones(M, 3).to(device),
                    ),
                    transforms=hp_estimator_nf,
                ).condition(observed_encodings),
                alpha_r_q_transform,
            ).to_event(1),
        )

        alpha = pyro.sample(
            "alpha",
            dist.Delta(alpha_r_q[..., 0]).to_event(1),
            infer={"is_auxiliary": True},
        )
        r = pyro.sample(
            "r",
            dist.Delta(alpha_r_q[..., 1]).to_event(1),
            infer={"is_auxiliary": True},
        )
        q = pyro.sample(
            "q",
            dist.Delta(alpha_r_q[..., 2]).to_event(1),
            infer={"is_auxiliary": True},
        )

        # coupling matrix A
        A_loc = pyro.param(
            name="A_loc",
            init_tensor=(
                initial_values["A_loc"]
                if initial_values["A_loc"] is not None
                else torch.zeros((C, M, M))
            ).to(device),
            event_dim=3,
        )
        A_scale = pyro.param(
            name="A_scale",
            init_tensor=(
                initial_values["A_scale"]
                if initial_values["A_scale"] is not None
                else 0.1 * torch.ones((C, M, M))
            ).to(device),
            constraint=constraints.positive,
            event_dim=3,
        )

        A = pyro.sample("A", dist.Normal(loc=A_loc, scale=A_scale).to_event(3))

        # latent activation X
        X_loc = pyro.param(
            name="X_loc",
            init_tensor=(
                initial_values["X_loc"]
                if initial_values["X_loc"] is not None
                else torch.zeros((M, T))
            ).to(device),
            event_dim=2,
        )
        X_scale = pyro.param(
            name="X_scale",
            init_tensor=(
                initial_values["X_scale"]
                if initial_values["X_scale"] is not None
                else 0.1 * torch.ones((M, T))
            ).to(device),
            constraint=constraints.positive,
            event_dim=2,
        )

        X_transposed = pyro.sample(
            "X_transposed",
            dist.Normal(
                loc=torch.transpose(X_loc, -1, -2),
                scale=torch.transpose(X_scale, -1, -2),
            ).to_event(2),
        )

        X = pyro.sample(
            "X",
            dist.Delta(torch.transpose(X_transposed, -1, -2)).to_event(2),
            infer={"is_auxiliary": True},
        )

        return {"A": A, "q": q, "X": X, "r": r, "alpha": alpha}

    return guide


def load_single_mf_guide(
    param_store_filepath: str,
    single_mf_guide_config: SingleMFGuideConfig,
    hp_estimator_nf: dist.transforms.ComposeTransformModule,
) -> Callable:
    """Returns trained variational guide

    Args:
        param_store_filepath (str)
        single_mf_guide_config (SingleMFGuideConfig): from hydra
        hp_estimator_nf (dist.transforms.ComposeTransformModule)

    Returns:
        Callable: variational guide
    """

    M = single_mf_guide_config["single_model_config"]["n_regions"]
    context_dim = hp_estimator_nf.parts[0]._modules["nn"].context_dim

    guide = get_single_mf_guide(single_mf_guide_config, hp_estimator_nf)

    guide(observed_encodings=torch.zeros(M, context_dim))

    param_state = torch.load(param_store_filepath)
    pyro.get_param_store().set_state(param_state)

    return guide
