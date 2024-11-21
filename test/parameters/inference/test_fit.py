import pytest

from mdshvb.parameters.inference.general import fit_guide_on_data


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "model_guide_data",
    [
        (
            "example_single_model",
            "example_single_mf_guide",
            "example_single_data_encoding",
        )
    ],
)
def test_fit(model_guide_data, example_inference_training_config, request):
    model, guide, observed_data = (
        request.getfixturevalue(fixture) for fixture in model_guide_data
    )

    fit_guide_on_data(
        model,
        guide,
        observed_data,
        training_config=example_inference_training_config,
    )
