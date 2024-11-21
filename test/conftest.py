from typing import Tuple

import pytest
from mdshvb.generative.hrf import HRFConfig
import torch
import pyro
from mdshvb.generative.single_bm import (
    FKLDatasetConfig,
    SingleModelConfig,
    CouplingMatrixPriorConfig,
    get_single_model,
)
from mdshvb.hyper_parameters.alpha_r_q import (
    BOLDScalePriorConfig,
    HRFAnglePriorConfig,
    LatentScalePriorConfig,
    NFConfig,
    NFEncoderConfig,
    EncoderConfig,
    get_nf_encoder,
)
from mdshvb.hyper_parameters.training import (
    HPEstimatorConfig,
    HPTrainingConfig,
)
from mdshvb.parameters.inference.general import InferenceTrainingConfig
from mdshvb.parameters.variational.single_mf_family import (
    SingleMFGuideConfig,
    SingleMFGuideHyperparametersConfig,
    get_single_mf_guide,
)


@pytest.fixture
def example_n_regions() -> int:
    return 3


@pytest.fixture
def example_n_timesteps() -> int:
    return 20


@pytest.fixture
def example_n_conditions() -> int:
    return 2


@pytest.fixture
def example_single_bm_config(
    example_n_regions, example_n_timesteps, example_n_conditions
) -> SingleModelConfig:
    return SingleModelConfig(
        n_regions=example_n_regions,
        n_timesteps=example_n_timesteps,
        n_conditions=example_n_conditions,
        coupling_matrix_prior_config=CouplingMatrixPriorConfig(
            type="Laplace",
            laplace_self_coupling_loc=0.8,
            laplace_self_coupling_scale=0.05,
            laplace_coupling_scale=0.1,
        ),
        latent_scale_prior_config=LatentScalePriorConfig(
            normal_loc=1.0, normal_scale=0.1
        ),
        bold_scale_prior_config=BOLDScalePriorConfig(
            normal_loc=1.0, normal_scale=0.1
        ),
        hrf_angle_prior_config=HRFAnglePriorConfig(
            uniform_low=-0.1, uniform_high=0.1
        ),
        hrf_config=HRFConfig(time_repetition=1.0, time_duration_kernel=4.0),
    )


@pytest.fixture
def example_dataset_config(example_single_bm_config) -> FKLDatasetConfig:
    return FKLDatasetConfig(
        single_model_config=example_single_bm_config,
        dataset_size=21,
        Y_max_abs_val=200.0,
        seed=1234,
    )


@pytest.fixture
def example_encoder_config() -> EncoderConfig:
    return EncoderConfig(
        conv1d_channel_kernel_strides=[(3, 2, 1)],
        avgpool1d_kernels=[2],
        flat_size=27,
    )


@pytest.fixture
def example_nf_config() -> NFConfig:
    return NFConfig(
        n_blocks=1,
        hidden_dims=[8],
    )


@pytest.fixture
def example_context_dim() -> int:
    return 3


@pytest.fixture
def example_nf_encoder_config(
    example_encoder_config, example_nf_config, example_context_dim
) -> NFEncoderConfig:
    return NFEncoderConfig(
        nf_config=example_nf_config,
        encoder_config=example_encoder_config,
        context_dim=example_context_dim,
    )


@pytest.fixture
def example_hp_estimator_config(
    example_dataset_config, example_nf_encoder_config
) -> HPEstimatorConfig:
    return HPEstimatorConfig(
        fkl_dataset_config=example_dataset_config,
        nf_encoder_config=example_nf_encoder_config,
        training=HPTrainingConfig(
            val_ratio=0.5,
            train_batch_size=5,
            val_batch_size=5,
            loss_upper_threshold=None,
            lr=0.001,
            epochs=2,
        ),
    )


@pytest.fixture
def example_nf_encoder(
    example_nf_encoder_config,
) -> Tuple[
    pyro.distributions.transforms.ComposeTransformModule, torch.nn.Sequential
]:
    nf, encoder = get_nf_encoder(example_nf_encoder_config)

    return nf, encoder


@pytest.fixture
def example_single_mf_guide_config(
    example_single_bm_config,
) -> SingleMFGuideConfig:
    return SingleMFGuideConfig(
        single_model_config=example_single_bm_config,
        hyper_parameters=SingleMFGuideHyperparametersConfig(),
    )


@pytest.fixture
def example_inference_training_config() -> InferenceTrainingConfig:
    return InferenceTrainingConfig(
        lr=0.01,
        betas=(0.95, 0.999),
        num_particles=2,
        max_plate_nesting=0,
        epochs=2,
    )


@pytest.fixture
def example_single_model(example_single_bm_config):
    return get_single_model(example_single_bm_config)


@pytest.fixture
def example_single_mf_guide(
    example_single_mf_guide_config, example_nf_encoder
):
    nf, _ = example_nf_encoder
    return get_single_mf_guide(
        single_mf_guide_config=example_single_mf_guide_config,
        hp_estimator_nf=nf,
    )


@pytest.fixture
def example_single_observed_data(example_n_timesteps, example_n_regions):

    torch.manual_seed(1234)

    return dict(
        observed_Y=torch.rand((example_n_regions, example_n_timesteps)),
        observed_c=torch.zeros(example_n_timesteps).long(),
    )


@pytest.fixture
def example_single_data_encoding(
    example_single_observed_data, example_n_regions, example_context_dim
):
    return dict(
        **example_single_observed_data,
        observed_encodings=torch.zeros(example_n_regions, example_context_dim),
    )
