import pyro.distributions.transforms as trans

from torch.distributions.utils import _sum_rightmost


class SoftclipTransform(trans.ComposeTransform):
    """
    Combination of Affine and Sigmoid transforms that emulates a soft clipping
    """

    def __init__(
        self, low: float, high: float, slope: float = 3.5, event_dim: int = 0
    ) -> None:
        loc = (low + high) / 2
        scale = high - low
        super().__init__(
            [
                trans.AffineTransform(
                    loc=0.0, scale=slope, event_dim=event_dim
                ),
                trans.SigmoidTransform(),
                trans.AffineTransform(
                    loc=-0.5, scale=1.0, event_dim=event_dim
                ),
                trans.AffineTransform(
                    loc=0.0, scale=scale, event_dim=event_dim
                ),
                trans.AffineTransform(loc=loc, scale=1.0, event_dim=event_dim),
            ]
        )


class BatchedStackTransform(trans.StackTransform):
    """
    Hacky fix for `StackTransform` assuming that the stacking operation
    is performed over batch and not event dims
    """

    def __init__(self, tseq, dim=0, cache_size=0, reinterpreted_batch_ndims=1):
        super().__init__(tseq, dim, cache_size)

        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def log_abs_det_jacobian(self, x, y):
        batch_log_abs_det_jacobian = super().log_abs_det_jacobian(x, y)

        return _sum_rightmost(
            batch_log_abs_det_jacobian, self.reinterpreted_batch_ndims
        )
