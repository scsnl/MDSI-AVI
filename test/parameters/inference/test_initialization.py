import torch

from mdshvb.parameters.inference.initialization import (
    initialize_coupling,
    initialize_latent_signal,
)


def test_initialize_latent_signal(
    example_n_regions, example_n_timesteps, example_single_bm_config
):

    M = example_n_regions
    T = example_n_timesteps
    B = 7

    Y1 = torch.zeros(T)
    X1 = initialize_latent_signal(
        Y1, single_model_config=example_single_bm_config
    )

    assert X1.shape == (T,)

    Y2 = torch.zeros(M, T)
    X2 = initialize_latent_signal(
        Y2, single_model_config=example_single_bm_config
    )

    assert X2.shape == (M, T)

    Y3 = torch.zeros(B, M, T)
    X3 = initialize_latent_signal(
        Y3, single_model_config=example_single_bm_config
    )

    assert X3.shape == (B, M, T)


def test_initialize_coupling(example_n_regions, example_n_timesteps):

    torch.manual_seed(1234)

    M = example_n_regions
    T = example_n_timesteps
    B = 7

    X1 = torch.rand((M, T))
    A1 = initialize_coupling(X1)

    assert A1.shape == (M, M)

    X2 = torch.rand((B, M, T))
    A2 = initialize_coupling(X2)

    assert A2.shape == (B, M, M)

    X3 = torch.rand((B, B, M, T))
    A3 = initialize_coupling(X3)

    assert A3.shape == (B, B, M, M)
