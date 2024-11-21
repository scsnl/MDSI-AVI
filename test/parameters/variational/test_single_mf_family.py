import pytest

import torch
import pyro

from mdshvb.parameters.variational.single_mf_family import get_single_mf_guide


@pytest.mark.order(2)
def test_build_shapes_nonan(
    example_nf_encoder, example_context_dim, example_single_mf_guide_config
):
    pyro.set_rng_seed(1234)

    nf, _ = example_nf_encoder
    model_config = example_single_mf_guide_config["single_model_config"]

    M = model_config["n_regions"]
    T = model_config["n_timesteps"]
    C = model_config["n_conditions"]

    context_dim = example_context_dim

    encodings = torch.zeros((M, context_dim))

    guide = get_single_mf_guide(
        single_mf_guide_config=example_single_mf_guide_config,
        hp_estimator_nf=nf,
    )

    sample = guide(observed_encodings=encodings)

    assert sample["A"].shape == (C, M, M)
    assert sample["alpha"].shape == (M,)
    assert sample["q"].shape == (M,)
    assert sample["X"].shape == (M, T)
    assert sample["r"].shape == (M,)

    assert torch.all(
        torch.tensor(
            [~torch.any(torch.isnan(value)) for value in sample.values()]
        )
    )

    B = 7
    with pyro.plate("samples", B, dim=-1):
        sample = guide(observed_encodings=encodings)

    assert sample["A"].shape == (B, C, M, M)
    assert sample["alpha"].shape == (B, M)
    assert sample["q"].shape == (B, M)
    assert sample["X"].shape == (B, M, T)
    assert sample["r"].shape == (B, M)

    assert torch.all(
        torch.tensor(
            [~torch.any(torch.isnan(value)) for value in sample.values()]
        )
    )
