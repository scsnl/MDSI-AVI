import torch

from mdshvb.transforms import SoftclipTransform


def test_soft_clip():
    low = -4.0
    high = 0.2
    soft_clipper = SoftclipTransform(low=low, high=high)

    x1 = torch.tensor([-5.0])
    y1 = soft_clipper(x1)

    assert y1 > low

    x2 = torch.tensor([2.0])
    y2 = soft_clipper(x2)

    assert y2 < high

    x3 = torch.tensor([0.0])
    y3 = soft_clipper(x3)

    assert y3.numpy() == (low + high) / 2

    # other tests were done via manual plotting
