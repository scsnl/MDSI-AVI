import torch

from pyro.ops.gaussian import (
    gaussian_tensordot,
    matrix_and_mvn_to_gaussian,
    mvn_to_gaussian,
    sequential_gaussian_filter_sample,
    sequential_gaussian_tensordot,
)

from pyro.distributions import constraints
from pyro.distributions.util import broadcast_shape
from pyro.distributions.hmm import HiddenMarkovModel


class LinearGaussianMM(HiddenMarkovModel):
    """
    Adapts Pyro's GaussianHMM to the simple case of a
    linear Gaussian Markov model
    """

    has_rsample = True
    arg_constraints = {}
    support = constraints.independent(constraints.real, 2)

    def __init__(
        self,
        initial_dist,
        transition_matrix,
        transition_dist,
        validate_args=None,
        duration=None,
    ):
        assert isinstance(
            initial_dist, torch.distributions.MultivariateNormal
        ) or (
            isinstance(initial_dist, torch.distributions.Independent)
            and isinstance(initial_dist.base_dist, torch.distributions.Normal)
        )
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(
            transition_dist, torch.distributions.MultivariateNormal
        ) or (
            isinstance(transition_dist, torch.distributions.Independent)
            and isinstance(
                transition_dist.base_dist, torch.distributions.Normal
            )
        )

        hidden_dim = transition_matrix.shape[-1]
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)

        shape = broadcast_shape(
            initial_dist.batch_shape + (1,),
            transition_matrix.shape[:-2],
            transition_dist.batch_shape,
        )
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (hidden_dim,)
        super().__init__(
            duration, batch_shape, event_shape, validate_args=validate_args
        )

        self.hidden_dim = hidden_dim
        self._init = mvn_to_gaussian(initial_dist).expand(self.batch_shape)
        self._trans = matrix_and_mvn_to_gaussian(
            transition_matrix, transition_dist
        ).to_gaussian()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Combine observation and transition factors.
        result = self._trans

        # Eliminate time dimension.
        result = sequential_gaussian_tensordot(
            result.expand(result.batch_shape)
        )

        # Combine initial factor.
        result = gaussian_tensordot(self._init, result, dims=self.hidden_dim)

        # Marginalize out final state.
        result = result.event_logsumexp()
        return result

    def rsample(self, sample_shape=torch.Size()):
        assert self.duration is not None
        sample_shape = torch.Size(sample_shape)
        trans = self._trans
        trans = trans.expand(trans.batch_shape[:-1] + (self.duration,))

        z = sequential_gaussian_filter_sample(self._init, trans, sample_shape)
        z = z[..., 1:, :]  # drop the initial hidden state

        return z
