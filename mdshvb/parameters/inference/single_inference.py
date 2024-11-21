from collections import defaultdict
from typing import Dict, Literal, TypedDict, Tuple, List

import torch
import pyro.distributions as dist

from mdshvb.generative.single_bm import SingleModelConfig, get_single_model
from mdshvb.parameters.inference.general import (
    InferenceTrainingConfig,
    fit_guide_on_data,
)
from mdshvb.parameters.inference.initialization import (
    initialize_coupling,
    initialize_latent_signal,
)
from mdshvb.parameters.variational.single_mf_family import (
    SingleMFGuideConfig,
    SingleMFGuideHyperparametersConfig,
    get_single_mf_guide,
)
from mdshvb.utils import repeat_to_shape


class SingleInferenceConfig(TypedDict):
    single_model_config: SingleModelConfig
    single_guide_type: Literal["MF",]
    initialize_latent: Literal["Wiener"] | None
    initialize_coupling: Literal["eye", "least-squares"] | None
    single_guide_hyperparameters_config: SingleMFGuideHyperparametersConfig
    training: InferenceTrainingConfig


def infer_single(
    hp_estimator_nf: dist.transforms.ComposeTransformModule,
    hp_estimator_encoder: torch.nn.Sequential,
    single_inference_config: SingleInferenceConfig,
    observed_Y_c: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Tuple[Dict, List[float]]:
    """Creates model, guide, computes initial values for parameters
    and performs inference

    Args:
        hp_estimator_nf (dist.transforms.ComposeTransformModule)
        hp_estimator_encoder (torch.nn.Sequential)
        single_inference_config (SingleInferenceConfig): from hydra
        observed_Y_c (Dict[str, torch.Tensor]): dict(
             observed_Y=BOLD, shape (n_regions,n_timesteps)
             observed_c=conditions, shape (n_timesteps,)
        )
        device (str, optional): Defaults to "cpu".

    Raises:
        ValueError: bad initialization
        NotImplementedError: bad guide type

    Returns:
        Tuple[Dict, List[float]]: trained param_store, train losses
    """

    single_model_config = single_inference_config["single_model_config"]
    model = get_single_model(single_model_config, device=device)

    initial_values = defaultdict(lambda: None)
    if single_inference_config["initialize_latent"] == "Wiener":
        initial_values["X_loc"] = initialize_latent_signal(
            observed_bold_signal=observed_Y_c["observed_Y"],
            single_model_config=single_model_config,
            device=device,
        )
    if single_inference_config["initialize_coupling"] == "eye":
        initial_values["A_loc"] = repeat_to_shape(
            (
                single_model_config["coupling_matrix_prior_config"][
                    "laplace_self_coupling_loc"
                ]
                if single_model_config["coupling_matrix_prior_config"]["type"]
                == "Laplace"
                else 0.7
            )
            * torch.eye(single_model_config["n_regions"]),
            (single_model_config["n_conditions"],),
            dim=0,
        )
    elif single_inference_config["initialize_coupling"] == "least-squares":
        if initial_values["X_loc"] is None:
            raise ValueError(
                "`initialize_latent` should be `Wiener` to use "
                "`least-squares` `initialize_coupling`"
            )
        initial_values["A_loc"] = repeat_to_shape(
            initialize_coupling(initial_values["X_loc"]),
            (single_model_config["n_conditions"],),
            dim=0,
        )

    guide_type = single_inference_config["single_guide_type"]
    if guide_type == "MF":
        guide = get_single_mf_guide(
            single_mf_guide_config=SingleMFGuideConfig(
                single_model_config=single_model_config,
                hyper_parameters=single_inference_config[
                    "single_guide_hyperparameters_config"
                ],
            ),
            hp_estimator_nf=hp_estimator_nf,
            initial_values=initial_values,
            device=device,
        )
    else:
        raise NotImplementedError(f"Guide type {guide_type} not implemented")

    hp_estimator_encoder.to(device)
    observed_encodings = {
        "observed_encodings": hp_estimator_encoder(
            observed_Y_c["observed_Y"].to(device)[:, None]
        ).detach()
    }
    observed_data = {
        rv: tensor.to(device)
        for rv, tensor in dict(**observed_Y_c, **observed_encodings).items()
    }

    param_store, losses = fit_guide_on_data(
        model,
        guide,
        observed_data,
        training_config=single_inference_config["training"],
    )

    return param_store, losses
