from typing import Dict, TypedDict

import numpy as np
import torch

from nilearn.glm.first_level import compute_regressor


class HRFConfig(TypedDict):
    time_repetition: float
    time_duration_kernel: float


def get_spm_derivatives_hrf(
    n_regions: int,
    time_repetition: float,
    time_duration_kernel: float,
    device: str = "cpu",
) -> Dict:
    """Outputs a dictionnary containing spm+derivative
    hemodynamic response function primitives.

    Args:
        n_regions (int)
        time_repetition (float): tr
        time_duration_kernel (float): duration (s) of HRF
        device (str, optional): Defaults to "cpu".

    Returns:
        Dict: dict(
            kernel=3D kernel,
            kernel_time_derivative=3D kernel,
            regressor=1D regressor,
            regressor_time_derivative=1D regressor,
            n_kernel_timesteps=duration in number of timesteps,
        )
    """
    onsets_durations_amplitudes = np.array([[0.0], [time_repetition], [1.0]])
    frame_times = np.arange(0.0, time_duration_kernel, time_repetition)
    n_kernel_timesteps = len(frame_times)  # size of HRF kernel

    hrf_regressors = compute_regressor(
        exp_condition=onsets_durations_amplitudes,
        hrf_model="spm + derivative",
        frame_times=frame_times,
    )
    reg_1 = hrf_regressors[0][:, 0]
    reg_2 = hrf_regressors[0][:, 1]

    k1 = torch.from_numpy(reg_1 / np.max(reg_1)).to(torch.float32)
    k2 = torch.from_numpy(reg_2 / np.max(reg_2)).to(torch.float32)

    kernel = torch.stack(
        [torch.diag(k * torch.ones(n_regions)) for k in torch.flip(k1, [0])],
        dim=-1,
    )
    kernel_time_derivative = torch.stack(
        [torch.diag(k * torch.ones(n_regions)) for k in torch.flip(k2, [0])],
        dim=-1,
    )

    return dict(
        kernel=kernel.to(device),
        kernel_time_derivative=kernel_time_derivative.to(device),
        regressor=k1,
        regressor_time_derivative=k2,
        n_kernel_timesteps=n_kernel_timesteps,
    )
