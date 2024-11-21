# %%

import os

from omegaconf import OmegaConf
from hydra import compose, initialize

import numpy as np

from mdshvb.generative.single_bm import get_single_model
from mdshvb.plotting import (
    plot_coupling_estimate,
    plot_signal_estimate,
    plot_signal_posterior,
)
from mdshvb.parameters.inference.initialization import initialize_latent_signal

# %%

initialize(config_path="../../conf")

# %%

single_model_config = OmegaConf.to_object(
    compose(
        config_name="experiment/hp_estimator_config/fkl_dataset_config/single_model_config/example_single_bm"
    ).experiment.hp_estimator_config.fkl_dataset_config.single_model_config
)

# %%

model = get_single_model(single_model_config=single_model_config)

# %%

synthetic_data = model()

# %%

plot_coupling_estimate(A=synthetic_data["A"].numpy(), lims=[-0.5, 0.5])

# %%

plot_signal_estimate(signal=synthetic_data["Y"].numpy())

# %%

data_dir = "/home/mind/lrouilla/dragostore/MDSI_Pyro/inputs/example"
os.makedirs(data_dir, exist_ok=True)

for rv, tensor in synthetic_data.items():
    np.save(f"{data_dir}/{rv}.npy", tensor.numpy())

# %%

initial_X = initialize_latent_signal(
    synthetic_data["Y"],
    single_model_config=single_model_config,
    hrf_alpha=synthetic_data["alpha"],
)

# %%

plot_signal_posterior(
    signal_samples=initial_X.numpy()[None],
    signal_ground_truth=synthetic_data["X"].numpy(),
)
# %%
