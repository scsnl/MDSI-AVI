import torch
import pyro

from mdshvb.generative.single_bm import (
    generate_fkl_dataset,
    get_single_model,
)


def test_build_shapes_nonan(example_single_bm_config):
    pyro.set_rng_seed(1234)

    M = example_single_bm_config["n_regions"]
    T = example_single_bm_config["n_timesteps"]
    C = example_single_bm_config["n_conditions"]

    model = get_single_model(example_single_bm_config)

    sample = model()

    assert sample["A"].shape == (C, M, M)
    assert sample["c"].shape == (T,)
    assert sample["alpha"].shape == (M,)
    assert sample["q"].shape == (M,)
    assert sample["X"].shape == (M, T)
    assert sample["r"].shape == (M,)
    assert sample["Y"].shape == (M, T)

    assert ~torch.any(torch.isnan(sample["Y"]))

    B = 7
    with pyro.plate("samples", B, dim=-1):
        sample = model()

    assert sample["A"].shape == (B, C, M, M)
    assert sample["c"].shape == (B, T)
    assert sample["alpha"].shape == (B, M)
    assert sample["q"].shape == (B, M)
    assert sample["X"].shape == (B, M, T)
    assert sample["r"].shape == (B, M)
    assert sample["Y"].shape == (B, M, T)

    assert ~torch.any(torch.isnan(sample["Y"]))


def test_generate_fkl_dataset(example_dataset_config):
    pyro.set_rng_seed(1234)

    T = example_dataset_config["single_model_config"]["n_timesteps"]
    B = example_dataset_config["dataset_size"]

    fkl_dataset = generate_fkl_dataset(example_dataset_config)

    assert fkl_dataset["Y"].shape == (B, 1, T)
    assert fkl_dataset["alpha_r_q"].shape == (B, 3)
