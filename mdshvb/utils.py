import torch
from typing import Callable, Iterable, Dict

import hydra
from hydra.types import RunMode

from pyro import poutine, render_model


def repeat_to_shape(
    x: torch.Tensor, target_shape: Iterable, dim: int
) -> torch.Tensor:
    """Inserts given shape at specified dim

    Args:
        x (torch.Tensor)
        target_shape (Iterable)
        dim (int)

    Returns:
        torch.Tensor: repeated x
    """

    out = x
    if dim < 0:
        for size in target_shape if dim < 0 else reversed(target_shape):
            out = torch.repeat_interleave(
                torch.unsqueeze(out, dim=dim), repeats=size, dim=dim
            )

    return out


def inspect_model(
    model: Callable,
    model_kwargs: Dict[str, torch.Tensor] | None = None,
    graph_to: str | None = None,
) -> None:
    """Inspects batch and event shapes, optionally draws graph

    Args:
        model (Callable): Pyro model/guide
        model_kwargs (Dict[str, torch.Tensor] | None, optional): input data. Defaults to None.
        graph_to (str | None, optional): filename. Defaults to None.
    """

    trace = poutine.trace(model).get_trace(**model_kwargs)
    trace.compute_log_prob()
    print(trace.format_shapes())
    if graph_to is not None:
        render_model(model, model_kwargs=model_kwargs, filename=graph_to)


def get_hydra_output_dir() -> str:
    """Depending on hydra being in RUN or MULTIRUN,
    outputs the dir where configs will be stored

    Returns:
        str: dir
    """
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    if hydra_config.mode == RunMode.RUN:
        output_dir = hydra_config.run.dir
    elif hydra_config.mode == RunMode.MULTIRUN:
        output_dir = f"{hydra_config.sweep.dir}/{hydra_config.sweep.subdir}"

    return output_dir
