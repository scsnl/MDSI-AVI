from omegaconf import DictConfig, OmegaConf
import hydra
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from mdshvb.hyper_parameters.loading import load_nf_encoder
from mdshvb.parameters.inference.single_inference import (
    SingleInferenceConfig,
    infer_single,
)

from mdshvb.utils import get_hydra_output_dir


@hydra.main(version_base=None, config_path="../conf")
def main(config: DictConfig) -> None:
    ENCODER_WEIGHTS_FILENAME = "encoder_state_dict.pt"
    NF_WEIGHTS_FILENAME = "nf_state_dict.pt"
    BOLD_TIMESERIES_FILENAME = "Y.npy"
    CONDITION_TIMESERIES_FILENAME = "c.npy"
    PYRO_PARAM_STORE_FILENAME = "param_store.pt"
    LOSS_PLOT_FILENAME = "inference_loss.pdf"

    hydra_output_dir = get_hydra_output_dir()
    os.makedirs(hydra_output_dir, exist_ok=True)

    experiment_config = OmegaConf.to_object(config.experiment)  # type: Dict

    # ? Load hyper-parameter estimator
    nf, encoder = load_nf_encoder(
        nf_weights_filepath=f"{config.hp_weights_dir}/{NF_WEIGHTS_FILENAME}",
        encoder_weights_filepath=f"{config.hp_weights_dir}/{ENCODER_WEIGHTS_FILENAME}",
        nf_encoder_config=experiment_config["hp_estimator_config"][
            "nf_encoder_config"
        ],
    )

    # ? Load observed data
    observed_Y_c = {
        f"observed_{rv}": torch.from_numpy(
            np.load(f"{config.input_data_dir}/{filename}")
        )
        .to(torch.float32 if rv == "Y" else torch.long)
        .to(config.device)
        .detach()
        for rv, filename in [
            ("Y", BOLD_TIMESERIES_FILENAME),
            ("c", CONDITION_TIMESERIES_FILENAME),
        ]
    }

    param_store, losses = infer_single(
        hp_estimator_nf=nf,
        hp_estimator_encoder=encoder,
        single_inference_config=SingleInferenceConfig(
            single_model_config=experiment_config["hp_estimator_config"][
                "fkl_dataset_config"
            ]["single_model_config"],
            single_guide_type=experiment_config["single_guide_type"],
            initialize_latent=experiment_config["initialize_latent"],
            initialize_coupling=experiment_config["initialize_coupling"],
            single_guide_hyperparameters_config=experiment_config[
                "single_guide_hyperparameters_config"
            ],
            training=experiment_config["training"],
        ),
        observed_Y_c=observed_Y_c,
        device=config.device,
    )

    plt.plot(losses, label="- ELBO")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("symlog")
    plt.legend()
    plt.savefig(
        f"{hydra_output_dir}/{LOSS_PLOT_FILENAME}", bbox_inches="tight"
    )

    state = param_store.get_state()
    for key, value in state["params"].items():
        state["params"][key] = value.cpu()

    torch.save(state, f"{hydra_output_dir}/{PYRO_PARAM_STORE_FILENAME}")


if __name__ == "__main__":
    main()
