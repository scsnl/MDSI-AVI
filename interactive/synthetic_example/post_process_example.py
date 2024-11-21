# %%
import os
from omegaconf import OmegaConf
from hydra import compose, initialize

import torch
import numpy as np
import pyro

from mdshvb.hyper_parameters.loading import load_nf_encoder
from mdshvb.parameters.variational.single_mf_family import (
    SingleMFGuideConfig,
    load_single_mf_guide,
)
from mdshvb.plotting import plot_coupling_posterior, plot_signal_posterior

# %%

initialize(config_path="../../conf")

# %%

experiment_config = OmegaConf.to_object(
    compose(config_name="experiment/example_single_inference").experiment
)  # type: Dict
HP_WEIGHTS_DIR = (
    "/home/mind/lrouilla/dragostore/MDSI_Pyro/outputs/2024-03-18/17-37-01"
)
ENCODER_WEIGHTS_FILEPATH = f"{HP_WEIGHTS_DIR}/encoder_state_dict.pt"
NF_WEIGHTS_FILEPATH = f"{HP_WEIGHTS_DIR}/nf_state_dict.pt"
PARAM_STORE_FILEPATH = "/home/mind/lrouilla/dragostore/MDSI_Pyro/outputs/example_inference/param_store.pt"
INPUT_DATA_DIR = "/home/mind/lrouilla/dragostore/MDSI_Pyro/inputs/example"

# %%

# ? Load HP estimator
nf, encoder = load_nf_encoder(
    nf_weights_filepath=NF_WEIGHTS_FILEPATH,
    encoder_weights_filepath=ENCODER_WEIGHTS_FILEPATH,
    nf_encoder_config=experiment_config["hp_estimator_config"][
        "nf_encoder_config"
    ],
)

# %%

# ? Load observed data
observed_rvs = [
    filename.split(".")[0] for filename in os.listdir(INPUT_DATA_DIR)
]
observed_data = {
    rv: np.load(f"{INPUT_DATA_DIR}/{rv}.npy") for rv in observed_rvs
}
observed_encodings = encoder(torch.from_numpy(observed_data["Y"][:, None]))

# %%

# ? Load P estimator
guide = load_single_mf_guide(
    param_store_filepath=PARAM_STORE_FILEPATH,
    single_mf_guide_config=SingleMFGuideConfig(
        single_model_config=experiment_config["hp_estimator_config"][
            "fkl_dataset_config"
        ]["single_model_config"],
        hyper_parameters=experiment_config[
            "single_guide_hyperparameters_config"
        ],
    ),
    hp_estimator_nf=nf,
)

# %%

B = 256
with pyro.plate("samples", B, dim=-1):
    sample = guide(observed_encodings=observed_encodings)

# %%

plot_coupling_posterior(
    A_samples=sample["A"].detach().numpy(),
    A_ground_truth=observed_data["A"],
)
# %%

plot_signal_posterior(
    signal_samples=sample["X"].detach().numpy(),
    signal_ground_truth=observed_data["X"],
)
# %%
