import torch
import pyro
from pyro.distributions import ConditionalTransformedDistribution, Normal

from mdshvb.hyper_parameters.alpha_r_q import (
    get_nf_encoder,
)


def test_build_shapes_nonan(example_n_timesteps, example_nf_encoder_config):
    torch.manual_seed(1234)
    pyro.set_rng_seed(1234)

    nf, encoder = get_nf_encoder(example_nf_encoder_config)

    B = 7

    Y = torch.zeros(B, 1, example_n_timesteps)
    encoding = encoder(Y)

    assert encoding.shape == (B, example_nf_encoder_config["context_dim"])
    assert ~torch.any(torch.isnan(encoding))

    conditional_dist = ConditionalTransformedDistribution(
        base_dist=Normal(loc=torch.zeros(B, 3), scale=torch.ones(B, 3)),
        transforms=nf,
    )

    sample = conditional_dist.condition(encoding).sample()

    assert sample.shape == (B, 3)
    assert ~torch.any(torch.isnan(sample))

    lp = conditional_dist.condition(encoding).log_prob(sample)

    assert lp.shape == (B,)
    assert ~torch.any(torch.isnan(lp))
