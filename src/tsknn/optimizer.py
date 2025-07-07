import itertools
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import BaseCrossValidator

from .tsknn import tsknn


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


METRICS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
}


FREQ_DEFAULTS: Dict[str, Tuple[int, int]] = {
    "D": (7, 7),
    "W": (4, 4),
    "M": (12, 12),
    "Q": (4, 4),
    "A": (1, 1),
}


def infer_freq(series: Any) -> Optional[str]:
    """Return the inferred frequency string for a pandas indexed series."""

    if hasattr(series, "index") and isinstance(series.index, pd.DatetimeIndex):
        freq = getattr(series.index, "freqstr", None)
        if freq is None:
            freq = pd.infer_freq(series.index)
        return freq
    return None


def freq_params(freq: str) -> Tuple[int, int]:
    """Return default ``lags`` and ``h`` values for ``freq``."""

    key = freq.upper()[0]
    return FREQ_DEFAULTS.get(key, (3, 12))


def optimize_params(
    X: Sequence[float],
    param_grid: Dict[str, Iterable[Any]],
    test_size: int = 12,
    metric: Callable[[Sequence[float], Sequence[float]], float] = rmse,
    method: str = "grid",
    n_iter: int = 20,
    random_state: Optional[int] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
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
    early_stopping_rounds : int, optional
        Stop search if no improvement is seen for this many iterations.

    Returns
    -------
    tuple
        ``(best_params, best_score)`` with the best parameter combination found
        and the corresponding score.

    Notes
    -----
    ``method`` can be ``"grid"`` (default), ``"random"`` or ``"bayes"``. When
    using ``"random"`` or ``"bayes"`` the ``n_iter`` argument controls the number
    of parameter sets evaluated.
    """

    if isinstance(metric, str):
        metric_func = METRICS.get(metric.lower())
        if metric_func is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_func = metric

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.values.squeeze()
    X = np.asarray(X)
    best_params = None
    best_score = np.inf

    rng = np.random.default_rng(random_state)

    param_names = list(param_grid.keys())
    param_values = [list(param_grid[name]) for name in param_names]

    def evaluate(params: Dict[str, Any]) -> float:
        params = {
            k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()
        }
        horizon = params.get("h", test_size)
        params["h"] = horizon
        lags = params.get("lags", 1)
        max_lag = max(lags) if hasattr(lags, "__iter__") else lags

        if len(X) <= test_size + max_lag:
            return np.inf

        train = X[:-test_size]
        X_pred = train[-max_lag:]
        model = tsknn(**params)
        try:
            model.fit(train)
            preds = model.predict(X_pred)
        except ValueError:
            return np.inf

        score = metric_func(X[-test_size:], preds[:test_size])
        return score

    search_space = list(itertools.product(*param_values))

    if method == "random":
        rng.shuffle(search_space)
        search_space = search_space[:n_iter]
    elif method == "bayes":
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        space = [
            Categorical(values, name=name)
            for name, values in zip(param_names, param_values)
        ]

        @use_named_args(space)
        def objective(**params):
            return evaluate(params)

        res = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            n_initial_points=min(n_iter, 10),
            random_state=random_state,
        )
        best_score = float(res.fun)
        best_params = {
            name: (val.item() if isinstance(val, np.generic) else val)
            for name, val in zip(param_names, res.x)
        }
        return best_params, best_score

    no_improve = 0
    for values in search_space:
        params = dict(zip(param_names, values))
        score = evaluate(params)
        if score < best_score:
            best_score = score
            best_params = params
            no_improve = 0
        else:
            no_improve += 1
        if early_stopping_rounds is not None and no_improve >= early_stopping_rounds:
            break

    return best_params, best_score


def cross_validate_params(
    X: Sequence[float],
    param_grid: Dict[str, Iterable[Any]],
    test_size: int = 12,
    n_splits: int = 3,
    metric: Callable[[Sequence[float], Sequence[float]], float] = rmse,
    method: str = "grid",
    n_iter: int = 20,
    random_state: Optional[int] = None,
    cv: Optional[BaseCrossValidator] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Optimize parameters using rolling origin cross-validation.

    Parameters
    ----------
    X : array-like
        Time series values.
    param_grid : dict
        Dictionary where keys are parameter names and values are iterables of
        parameter settings to try.
    test_size : int, optional
        Size of each validation fold.
    n_splits : int, optional
        Number of cross-validation folds. Defaults to 3.
    metric : str or callable, optional
        Metric to evaluate predictions. Same options as ``optimize_params``.
    method : {"grid", "random", "bayes"}, optional
        Search strategy. Same options as ``optimize_params``.
    n_iter : int, optional
        Number of parameter sets evaluated for ``random`` or ``bayes`` search.
    random_state : int, optional
        Seed for randomization in ``random`` or ``bayes`` search.
    cv : BaseCrossValidator, optional
        scikit-learn cross-validator object to generate train/test splits.
    early_stopping_rounds : int, optional
        Stop search if no improvement is seen for this many iterations.

    Returns
    -------
    tuple
        ``(best_params, best_score)`` with the best parameter combination found
        and the mean score across folds.
    """

    if isinstance(metric, str):
        metric_func = METRICS.get(metric.lower())
        if metric_func is None:
            raise ValueError(f"Unknown metric '{metric}'")
    else:
        metric_func = metric

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.values.squeeze()
    X = np.asarray(X)

    rng = np.random.default_rng(random_state)

    param_names = list(param_grid.keys())
    param_values = [list(param_grid[name]) for name in param_names]

    def evaluate(params: Dict[str, Any]) -> float:
        params = {
            k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()
        }
        horizon = params.get("h", test_size)
        params["h"] = horizon
        lags = params.get("lags", 1)
        max_lag = max(lags) if hasattr(lags, "__iter__") else lags

        if cv is not None:
            splits = list(cv.split(X))
            required_len = max(max(train_idx) + 1 for train_idx, _ in splits)
            if len(X) <= required_len:
                return np.inf
        else:
            splits = []
            for split in range(n_splits):
                train_end = len(X) - (n_splits - split) * test_size
                train_idx = np.arange(train_end)
                test_idx = np.arange(train_end, train_end + test_size)
                splits.append((train_idx, test_idx))
            required_len = n_splits * test_size + max_lag
            if len(X) <= required_len:
                return np.inf

        scores = []
        for train_idx, test_idx in splits:
            train = X[train_idx]
            test = X[test_idx]
            if len(train) <= max_lag:
                return np.inf
            model = tsknn(**params)
            try:
                model.fit(train)
                preds = model.predict(train[-max_lag:])
            except ValueError:
                return np.inf

            fold_score = metric_func(test[:horizon], preds[:horizon])
            scores.append(fold_score)

        return float(np.mean(scores))

    search_space = list(itertools.product(*param_values))

    best_params = None
    best_score = np.inf

    if method == "random":
        rng.shuffle(search_space)
        search_space = search_space[:n_iter]
    elif method == "bayes":
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        space = [
            Categorical(values, name=name)
            for name, values in zip(param_names, param_values)
        ]

        @use_named_args(space)
        def objective(**params: Any) -> float:
            return evaluate(params)

        res = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            n_initial_points=min(n_iter, 10),
            random_state=random_state,
        )
        best_score = float(res.fun)
        best_params = {
            name: (val.item() if isinstance(val, np.generic) else val)
            for name, val in zip(param_names, res.x)
        }
        return best_params, best_score

    no_improve = 0
    for values in search_space:
        params = dict(zip(param_names, values))
        score = evaluate(params)
        if score < best_score:
            best_score = score
            best_params = params
            no_improve = 0
        else:
            no_improve += 1
        if early_stopping_rounds is not None and no_improve >= early_stopping_rounds:
            break

    return best_params, best_score


def autotsknn(
    X: Sequence[float] | pd.Series,
    k_values: Optional[Iterable[int]] = None,
    lags_values: Optional[Iterable[int]] = None,
    test_size: Optional[int] = None,
    metric: str | Callable[[Sequence[float], Sequence[float]], float] = "rmse",
    freq: Optional[str] = None,
    search_method: str = "grid",
    n_iter: int = 20,
    random_state: Optional[int] = None,
    early_stopping_rounds: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[tsknn, Dict[str, Any], float]:
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
        If ``freq`` is provided and ``test_size`` is ``None``, a default value
        based on the frequency is used.
    freq : str, optional
        Frequency string (e.g. ``"D"``, ``"W"``, ``"M"``) used to infer default
        ``lags`` and ``h`` values when they are not supplied.
    metric : str or callable, optional
        Metric used to evaluate predictions. Same options as ``optimize_params``.
    search_method : {"grid", "random", "bayes"}, optional
        Optimization strategy used when searching parameters.
    n_iter : int, optional
        Number of parameter sets evaluated for ``random`` or ``bayes`` search.
    random_state : int, optional
        Seed used for randomization in ``random`` or ``bayes`` search.
    early_stopping_rounds : int, optional
        Stop search if no improvement is seen for this many iterations.
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

    if isinstance(X, (pd.Series, pd.DataFrame)):
        if freq is None:
            freq = infer_freq(X)
        X_values = X.values.squeeze()
    else:
        X_values = np.asarray(X)

    if freq and lags_values is None:
        l_default, h_default = freq_params(freq)
        lags_values = [l_default]
        kwargs.setdefault("h", h_default)
        if test_size is None:
            test_size = h_default

    if k_values is None:
        k_values = range(1, 6)
    if lags_values is None:
        lags_values = range(1, 4)

    if test_size is None:
        test_size = 12

    h = kwargs.pop("h", 12)
    param_grid = {"k": k_values, "lags": lags_values, "h": [h]}
    best_params, best_score = optimize_params(
        X_values,
        param_grid,
        test_size=test_size,
        metric=metric,
        method=search_method,
        n_iter=n_iter,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    if best_params is None:
        raise ValueError("No valid parameter combination found")

    best_params.update(kwargs)
    model = tsknn(**best_params)
    model.fit(X_values)
    return model, best_params, best_score
