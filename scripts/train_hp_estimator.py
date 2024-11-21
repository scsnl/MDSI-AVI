from omegaconf import DictConfig, OmegaConf
import hydra
import os

import torch
import matplotlib.pyplot as plt

from mdshvb.hyper_parameters.training import train_nf_encoder
from mdshvb.utils import get_hydra_output_dir


@hydra.main(version_base=None, config_path="../conf")
def main(config: DictConfig) -> None:
    ENCODER_WEIGHTS_FILENAME = "encoder_state_dict.pt"
    NF_WEIGHTS_FILENAME = "nf_state_dict.pt"
    LOSS_PLOT_FILENAME = "sbi_loss.pdf"

    output_dir = get_hydra_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    nf, encoder, losses = train_nf_encoder(
        hp_estimator_config=OmegaConf.to_object(
            config.experiment.hp_estimator_config
        ),
        device=config.device,
    )

    torch.save(
        encoder.cpu().state_dict(), f"{output_dir}/{ENCODER_WEIGHTS_FILENAME}"
    )
    torch.save(nf.cpu().state_dict(), f"{output_dir}/{NF_WEIGHTS_FILENAME}")

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.xlabel("Epochs")
    plt.ylabel("- forward KL")
    plt.yscale("symlog")
    plt.legend()
    plt.savefig(f"{output_dir}/{LOSS_PLOT_FILENAME}", bbox_inches="tight")


if __name__ == "__main__":
    main()
