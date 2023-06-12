import torch


def test_gpu():
    assert torch.cuda.is_available()

    x = torch.randn(2, 3).cuda()
    y = torch.randn(2, 3).cuda()
    z = x + y
    assert z.shape == (2, 3)
