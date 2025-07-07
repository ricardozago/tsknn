from dataclasses import FrozenInstanceError
import pytest

from tsknn.tsknn import TSKNNConfig


def test_config_is_frozen():
    cfg = TSKNNConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.k = 5
