import torch

from functools import partial
from mdshvb.generative.single_bm import SingleModelConfig
from mdshvb.generative.hrf import get_spm_derivatives_hrf
from mdshvb.utils import repeat_to_shape


def initialize_latent_signal(
    observed_bold_signal: torch.Tensor,
    single_model_config: SingleModelConfig,
    hrf_alpha: torch.Tensor | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Performs Wiener deconvolution

    Args:
        observed_bold_signal (torch.Tensor): batch + (n_regions, n_timesteps)
        single_model_config (SingleModelConfig): from hydra
        hrf_alpha (torch.Tensor, optional) for convolution kernel
            shape batch
        device (str, optional): Defaults to "cpu".

    Returns:
        torch.Tensor: latent signal, batch + (n_regions, n_timesteps)
    """

    M = single_model_config["n_regions"]
    T = single_model_config["n_timesteps"]
    tr = single_model_config["hrf_config"]["time_repetition"]

    if hrf_alpha is None:
        hrf_alpha = (
            torch.tensor(
                single_model_config["hrf_angle_prior_config"]["uniform_low"]
                + single_model_config["hrf_angle_prior_config"]["uniform_high"]
            )
            / 2
        )

    hrf_dict = get_spm_derivatives_hrf(
        n_regions=M,
        time_repetition=single_model_config["hrf_config"]["time_repetition"],
        time_duration_kernel=single_model_config["hrf_config"][
            "time_duration_kernel"
        ],
    )

    mean_hrf_kernel = (
        torch.cos(hrf_alpha)[..., None] * hrf_dict["regressor"]
        + torch.sin(hrf_alpha)[..., None]
        * hrf_dict["regressor_time_derivative"]
    )
    K = hrf_dict["n_kernel_timesteps"]

    B = torch.fft.fft(
        torch.cat(
            [
                mean_hrf_kernel,
                torch.zeros(*(mean_hrf_kernel.shape[:-1] + (T - K,))),
            ],
            dim=-1,
        )
    )

    self_coupling = (
        single_model_config["coupling_matrix_prior_config"][
            "laplace_self_coupling_loc"
        ]
        if single_model_config["coupling_matrix_prior_config"]["type"]
        == "Laplace"
        else 0.7
    )
    G = (
        -self_coupling + torch.exp(-1.0j + torch.fft.fftfreq(n=T, d=tr))
    ) ** -1

    # ? power for the convolution
    S_b = torch.abs(B) ** 2
    # ? power for the system response
    S_x = (
        single_model_config["latent_scale_prior_config"]["normal_loc"] ** 2
    ) * (torch.abs(G) ** 2)
    # ? power for the BOLD noise
    S_e = single_model_config["bold_scale_prior_config"]["normal_loc"] ** 2

    # ? total bold signal power
    S_signal = S_x * S_b
    # ? BOLD noise power
    S_noise = S_e

    # ? S_signal >> S_noise, phi goes to 1...
    phi = S_signal / (S_signal + S_noise)  # (Nf)

    Y_fft = torch.fft.fft(observed_bold_signal)
    X_fft = (
        repeat_to_shape(phi * 1 / B, Y_fft.shape[:-1], dim=0).to(device)
        * Y_fft
    )

    X = torch.real(torch.fft.ifft(X_fft))

    return X.float()


def initialize_coupling(latent_signal: torch.Tensor) -> torch.Tensor:
    """Uses least squares regression

    Args:
        latent_signal (torch.Tensor): batch + (n_regions, n_timesteps)

    Returns:
        torch.Tensor: batch + (n_regions, n_regions)
    """

    X_past = latent_signal[..., :-1]
    X_future = latent_signal[..., 1:]

    evidence = torch.einsum("...mt,...nt->...mn", X_past, X_past)

    p_inv = torch.pinverse(
        torch.einsum(
            "...mn,...nt->...mt",
            torch.inverse(evidence),
            X_past,
        )
    )
    A = torch.einsum("...tm,...nt->...mn", p_inv, X_future)

    A_transposed = torch.transpose(A, -1, -2)

    return A_transposed.float()
