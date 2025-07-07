import numpy as np
from tsknn import tsknn, mtsknn


def test_save_load_tsknn(tmp_path):
    X = np.arange(10, dtype=float)
    model = tsknn(lags=2, h=1)
    model.fit(X)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = tsknn.load(path)
    assert np.allclose(model.predict(X[-2:]), loaded.predict(X[-2:]))


def test_save_load_mtsknn(tmp_path):
    X = np.arange(20, dtype=float).reshape(-1, 2)
    model = mtsknn(lags=2, h=1)
    model.fit(X)
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = mtsknn.load(path)
    assert np.allclose(model.predict(X[-2:]), loaded.predict(X[-2:]))
