import pytest

from mdshvb.hyper_parameters.training import train_nf_encoder


@pytest.mark.order(1)
def test_run(example_hp_estimator_config):
    train_nf_encoder(example_hp_estimator_config)
