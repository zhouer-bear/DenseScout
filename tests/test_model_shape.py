import torch
from densescout.model import DenseScout


def test_model_shape_cpu():
    model = DenseScout(pretrained=False).eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 3, 640, 640))
    assert list(y.shape) == [1, 1, 80, 80]
