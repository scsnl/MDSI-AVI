from typing import Tuple, TypedDict, Dict, List
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pyro.distributions as dist
from tqdm import tqdm

from mdshvb.generative.single_bm import FKLDatasetConfig, generate_fkl_dataset
from mdshvb.hyper_parameters.alpha_r_q import (
    NFEncoderConfig,
    get_nf_encoder,
    get_transform,
)


class HPTrainingConfig(TypedDict):
    val_ratio: float
    train_batch_size: int
    val_batch_size: int
    loss_upper_threshold: float | None
    lr: float
    epochs: int


class HPEstimatorConfig(TypedDict):
    fkl_dataset_config: FKLDatasetConfig
    nf_encoder_config: NFEncoderConfig
    training: HPTrainingConfig


class FKLDataset(Dataset):
    def __init__(self, Y: torch.Tensor, alpha_r_q: torch.Tensor) -> None:
        super().__init__()

        self.Y = Y
        self.alpha_r_q = alpha_r_q

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.Y[index], self.alpha_r_q[index]


def split_data(Y, alpha_r_q, val_ratio=0.05) -> Tuple[FKLDataset, FKLDataset]:
    n_data = len(Y)
    n_data_val = int(n_data * val_ratio)

    perm = torch.randperm(n_data)
    Y_train = Y[perm][n_data_val:]
    Y_val = Y[perm][:n_data_val]

    alpha_r_q_train = alpha_r_q[perm][n_data_val:]
    alpha_r_q_val = alpha_r_q[perm][:n_data_val]

    train_dataset = FKLDataset(Y_train, alpha_r_q_train)
    val_dataset = FKLDataset(Y_val, alpha_r_q_val)

    return train_dataset, val_dataset


def train_nf_encoder(
    hp_estimator_config: HPEstimatorConfig,
    device: str = "cpu",
) -> Tuple[
    dist.transforms.ComposeTransformModule,
    torch.nn.Sequential,
    Dict[str, List[float]],
]:
    """Performs f-KL training of the HP estimator over a synthetic dataset

    Args:
        hp_estimator_config (HPEstimatorConfig): from hydra
        device (str, optional): Defaults to "cpu".

    Returns:
        Tuple[ dist.transforms.ComposeTransformModule, torch.nn.Sequential, Dict[str, List[float]], ]:
            nf, encoder, dict of `train` and `val` losses
    """
    fkl_dataset = generate_fkl_dataset(
        hp_estimator_config["fkl_dataset_config"]
    )

    training_config = hp_estimator_config["training"]
    train_dataset, val_dataset = split_data(
        fkl_dataset["Y"],
        fkl_dataset["alpha_r_q"],
        val_ratio=training_config["val_ratio"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=training_config["val_batch_size"]
    )

    nf, encoder = get_nf_encoder(hp_estimator_config["nf_encoder_config"])
    nf.to(device)
    encoder.to(device)

    single_model_config = hp_estimator_config["fkl_dataset_config"][
        "single_model_config"
    ]
    alpha_r_q_transform = get_transform(
        hrf_angle_prior_config=single_model_config["hrf_angle_prior_config"],
        bold_scale_prior_config=single_model_config["bold_scale_prior_config"],
        latent_scale_prior_config=single_model_config[
            "latent_scale_prior_config"
        ],
    )

    base_dist = dist.Normal(
        loc=torch.zeros(3).to(device),
        scale=torch.ones(3).to(device),
    )
    transformed_dist = dist.ConditionalTransformedDistribution(
        base_dist, transforms=nf
    )

    modules = torch.nn.ModuleList([nf, encoder])
    optimizer = torch.optim.Adam(
        modules.parameters(), lr=training_config["lr"]
    )

    epochs = training_config["epochs"]

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_encoder_state_dict = None
    best_nf_state_dict = None

    pbar_epochs = tqdm([e for e in range(epochs)], position=0, leave=True)

    for _ in pbar_epochs:
        pbar_batch = tqdm(train_dataloader, position=1, leave=False)

        modules.train()
        loss_acc = []
        for Y_batch, alpha_r_q_batch in pbar_batch:
            try:
                optimizer.zero_grad()
                encoding = encoder(
                    Y_batch.detach().type(torch.float32).to(device)
                )
                conditioned_dist = dist.TransformedDistribution(
                    transformed_dist.condition(encoding),
                    alpha_r_q_transform,
                )
                lp_alpha_r_q = conditioned_dist.log_prob(
                    alpha_r_q_batch.detach().type(torch.float32).to(device)
                )
                loss = -lp_alpha_r_q.mean()
                if (
                    loss < training_config["loss_upper_threshold"]
                    if training_config["loss_upper_threshold"] is not None
                    else True
                ):  # avoid catastrophic batches
                    loss.backward()
                    loss_acc.append(loss.detach().cpu().numpy())

                    optimizer.step()
                    transformed_dist.clear_cache()

                pbar_batch.set_description(
                    f"- Forward KL: {loss.detach().cpu().numpy():<2.4f}"
                )
            except ValueError:
                pass

        train_loss = np.mean(loss_acc)
        train_losses.append(train_loss)

        modules.eval()
        with torch.no_grad():
            loss_acc = []
            for Y_batch, alpha_r_q_batch in val_dataloader:
                try:
                    encoding = encoder(
                        Y_batch.detach().type(torch.float32).to(device)
                    )
                    conditioned_dist = dist.TransformedDistribution(
                        transformed_dist.condition(encoding),
                        alpha_r_q_transform,
                    )
                    lp_alpha_r_q = conditioned_dist.log_prob(
                        alpha_r_q_batch.detach().type(torch.float32).to(device)
                    )
                    loss = -lp_alpha_r_q.mean()
                    loss_acc.append(loss.detach().cpu().numpy())
                except ValueError:
                    pass

            new_val_loss = np.mean(loss_acc)
            val_losses.append(new_val_loss)

            if new_val_loss < best_val_loss:
                best_val_loss = new_val_loss
                best_encoder_state_dict = deepcopy(encoder.state_dict())
                best_nf_state_dict = deepcopy(nf.state_dict())

        pbar_epochs.set_description(f"- Forward KL: {best_val_loss:<2.4f}")

    encoder.load_state_dict(best_encoder_state_dict)
    nf.load_state_dict(best_nf_state_dict)

    return nf, encoder, dict(train=train_losses, val=val_losses)
