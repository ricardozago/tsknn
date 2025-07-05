import itertools
import numpy as np

from .tsknn import tsknn


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def optimize_params(X, param_grid, test_size=12, metric=rmse):
    """Optimize tsknn parameters for a single time series.

    Parameters
    ----------
    X : array-like
        Time series values.
    param_grid : dict
        Dictionary where keys are parameter names and values are iterables of
        parameter settings to try.
    test_size : int, optional
        Number of observations at the end of ``X`` to use as validation set.
    metric : callable, optional
        Metric function with signature ``metric(y_true, y_pred)``. Defaults to
        RMSE.

    Returns
    -------
    tuple
        ``(best_params, best_score)`` with the best parameter combination found
        and the corresponding score.
    """

    X = np.asarray(X)
    best_params = None
    best_score = np.inf

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        horizon = params.get("h", test_size)
        params["h"] = horizon
        lags = params.get("lags", 1)
        max_lag = max(lags) if hasattr(lags, "__iter__") else lags

        if len(X) <= test_size + max_lag:
            continue

        train = X[:-test_size]
        X_pred = train[-max_lag:]
        model = tsknn(**params)
        try:
            model.fit(train)
            preds = model.predict(X_pred)
        except ValueError:
            # skip invalid parameter combinations
            continue

        score = metric(X[-test_size:], preds[:test_size])
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score
