from typing import Callable, Dict, List, Tuple, TypedDict

import torch
import pyro
from pyro.optim import AdamW
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm


class InferenceTrainingConfig(TypedDict):
    lr: float
    betas: Tuple[float, float]
    num_particles: int
    max_plate_nesting: int
    epochs: int


def fit_guide_on_data(
    model: Callable,
    guide: Callable,
    observed_data: Dict[str, torch.Tensor],
    training_config: InferenceTrainingConfig,
) -> Tuple[Dict, List[float]]:
    """Trains the parameter estimator using the r-KL (ELBO)

    Args:
        model (Callable)
        guide (Callable)
        observed_data (Dict[str, torch.Tensor])
        training_config (InferenceTrainingConfig): from hydra

    Returns:
        Tuple[Dict, List[float]]: trined param_store, train losses
    """
    optim = AdamW(
        dict(lr=training_config["lr"], betas=training_config["betas"])
    )
    loss = Trace_ELBO(
        num_particles=training_config["num_particles"],
        max_plate_nesting=training_config["max_plate_nesting"],
        vectorize_particles=True,
    )

    svi = SVI(model, guide, optim, loss)

    pbar = tqdm([s for s in range(training_config["epochs"])])
    losses = []
    for _ in pbar:
        step_loss = svi.step(**observed_data)
        pbar.set_description(f"- ELBO: {step_loss: >50.0f}")
        losses.append(step_loss)

    param_store = pyro.get_param_store()

    return param_store, losses
