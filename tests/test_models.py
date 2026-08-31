import torch

from config import Config
from models import get_model


def test_autoencoder_forward_shape_roundtrip():
    model = get_model()
    x = torch.randn(8, Config.SEQ_LENGTH, Config.NUM_FEATURES)
    out = model(x)
    assert out.shape == x.shape


def test_autoencoder_is_opacus_compatible():
    """
    The model must stay Opacus-compatible (no BatchNorm, no unsupported
    layers) since local training runs under DP-SGD for every client.
    """
    from opacus.validators import ModuleValidator

    model = get_model()
    errors = ModuleValidator.validate(model, strict=False)
    assert errors == []
