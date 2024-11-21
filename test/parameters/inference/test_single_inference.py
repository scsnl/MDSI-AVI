import pytest
from contextlib import nullcontext

from mdshvb.parameters.inference.single_inference import (
    SingleInferenceConfig,
    infer_single,
)
from mdshvb.parameters.variational.single_mf_family import (
    SingleMFGuideHyperparametersConfig,
)


@pytest.mark.parametrize(
    "guide_type,guide_config", [("MF", SingleMFGuideHyperparametersConfig())]
)
@pytest.mark.parametrize("initialize_latent", ["Wiener", None])
@pytest.mark.parametrize("initialize_coupling", ["eye", "least-squares", None])
def test_inference(
    guide_type,
    guide_config,
    initialize_latent,
    initialize_coupling,
    example_single_bm_config,
    example_nf_encoder,
    example_inference_training_config,
    example_single_observed_data,
):

    nf, encoder = example_nf_encoder

    single_inference_config = SingleInferenceConfig(
        single_model_config=example_single_bm_config,
        single_guide_type=guide_type,
        initialize_latent=initialize_latent,
        initialize_coupling=initialize_coupling,
        single_guide_hyperparameters_config=guide_config,
        training=example_inference_training_config,
    )

    with (
        pytest.raises(ValueError)
        if (
            initialize_coupling == "least-squares"
            and initialize_latent != "Wiener"
        )
        else nullcontext()
    ):
        infer_single(
            hp_estimator_nf=nf,
            hp_estimator_encoder=encoder,
            single_inference_config=single_inference_config,
            observed_Y_c=example_single_observed_data,
        )
