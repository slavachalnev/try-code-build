import torch

def test_cpu():
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    assert z.shape == (2, 3)

