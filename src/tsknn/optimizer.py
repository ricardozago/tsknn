import itertools
import numpy as np

from .tsknn import tsknn


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true))


METRICS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
}


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
    metric : str or callable, optional
        Metric to evaluate predictions. Can be one of ``"rmse"``, ``"mae"`` or
        ``"mape"`` or a callable with signature ``metric(y_true, y_pred)``.
        Defaults to RMSE.

    Returns
    -------
    tuple
        ``(best_params, best_score)`` with the best parameter combination found
        and the corresponding score.
    """

    if isinstance(metric, str):
        metric_func = METRICS.get(metric.lower())
        if metric_func is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_func = metric

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

        score = metric_func(X[-test_size:], preds[:test_size])
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def autotsknn(X, k_values=None, lags_values=None, test_size=12, metric="rmse", **kwargs):
    """Find and fit the best tsknn model over ranges of ``k`` and ``lags``.

    Parameters
    ----------
    X : array-like
        Time series values.
    k_values : iterable of int, optional
        Values of ``k`` (number of neighbors) to try. Defaults to ``range(1, 6)``.
    lags_values : iterable of int, optional
        Values of ``lags`` to try. Defaults to ``range(1, 4)``.
    test_size : int, optional
        Number of observations to hold out from the end of ``X`` for validation.
    metric : str or callable, optional
        Metric used to evaluate predictions. Same options as ``optimize_params``.
    **kwargs
        Additional parameters passed to ``tsknn``.

    Returns
    -------
    model : :class:`tsknn.tsknn`
        Fitted model using the best combination of parameters.
    best_params : dict
        Parameter set that achieved the best score.
    best_score : float
        Score obtained for ``best_params``.
    """

    if k_values is None:
        k_values = range(1, 6)
    if lags_values is None:
        lags_values = range(1, 4)

    h = kwargs.pop("h", 12)
    param_grid = {"k": k_values, "lags": lags_values, "h": [h]}
    best_params, best_score = optimize_params(X, param_grid, test_size=test_size, metric=metric)

    if best_params is None:
        raise ValueError("No valid parameter combination found")

    best_params.update(kwargs)
    model = tsknn(**best_params)
    model.fit(X)
    return model, best_params, best_score
