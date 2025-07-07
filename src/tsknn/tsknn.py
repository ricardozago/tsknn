"""Core KNN utilities for time series forecasting."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, RegressorMixin

np.set_printoptions(suppress=True)
from statsmodels.tsa.stattools import pacf


def sum_euclidean(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the squared Euclidean distance between ``v`` and each row of ``M``."""

    # https://stackoverflow.com/a/49633639
    tmp = M - v
    return np.einsum("ij,ij->i", tmp, tmp)


def sum_manhattan(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the Manhattan distance between ``v`` and each row of ``M``."""

    tmp = M - v
    return np.einsum("ij->i", np.abs(tmp))


def max_chebyshev(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the Chebyshev distance between ``v`` and each row of ``M``."""

    tmp = M - v
    return np.max(np.abs(tmp), axis=1)


def cosine_distance(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the cosine distance between ``v`` and each row of ``M``."""

    dot_prod = np.einsum("ij,j->i", M, v)
    norm_M = np.linalg.norm(M, axis=1)
    norm_v = np.linalg.norm(v)
    denom = norm_M * norm_v
    denom = np.where(denom == 0, 1e-10, denom)
    return 1 - dot_prod / denom


def dtw_distance(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the DTW distance between ``v`` and each row of ``M``."""

    def _dtw(x: np.ndarray, y: np.ndarray) -> float:
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )
        return float(dtw_matrix[n, m])

    return np.array([_dtw(row, v) for row in M])


def get_distance(
    distance: str = "euclidean",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a distance function identified by ``distance``."""

    if distance == "euclidean":
        return sum_euclidean
    if distance == "manhattan":
        return sum_manhattan
    if distance == "chebyshev":
        return max_chebyshev
    if distance == "cosine":
        return cosine_distance
    if distance == "dtw":
        return dtw_distance
    return sum_euclidean


def select_lags_pacf(
    x: Sequence[float], nlags: int, threshold: float = 0.2
) -> List[int]:
    """Return lag indices with partial autocorrelation above ``threshold``."""

    pacf_vals = pacf(x, nlags=nlags)
    lags = [i for i, val in enumerate(pacf_vals[1:], start=1) if abs(val) >= threshold]
    return lags if lags else list(range(1, nlags + 1))


@dataclass
class TSKNNConfig:
    """Configuration options for :class:`tsknn`."""

    k: int | str | Sequence[int] = 3
    cf: str = "mean"
    transform: Optional[str] = None
    lags: int | Sequence[int] = 3
    distance: str = "euclidean"
    h: int = 12
    msas: str = "recursive"
    kmeans: Optional[int] = None
    random_state: Optional[int] = None
    nan_strategy: str = "propagate"
    weight_by: str = "recency"


class tsknn(BaseEstimator, RegressorMixin):
    """K-nearest neighbors forecasting for univariate time series."""

    def __init__(
        self,
        k: int | str | Sequence[int] = 3,
        cf: str = "mean",
        transform: Optional[str] = None,  # "additive", "multiplicative"
        lags: int | Sequence[int] = 3,
        distance: str = "euclidean",
        h: int = 12,
        msas: str = "recursive",
        kmeans: Optional[int] = None,
        random_state: Optional[int] = None,
        nan_strategy: str = "propagate",
        weight_by: str = "recency",
    ) -> None:
        """Initialize a ``tsknn`` model.

        Parameters
        ----------
        k : int or str or iterable, optional
            Number of neighbors or strategy to determine ``k``. Defaults to 3.
        cf : {"mean", "median", "weighted", "trimmed"}, optional
            Aggregation function used to combine neighbors.
        transform : {"additive", "multiplicative"}, optional
            Pre-processing transformation applied before computing distances.
        lags : int or iterable, optional
            Lag values to use when constructing the feature matrix.
        distance : {"euclidean", "manhattan", "chebyshev", "cosine"}, optional
            Distance metric to use. Defaults to Euclidean.
        h : int, optional
            Forecast horizon. Defaults to 12.
        msas : {"recursive", "mimo", "direct"}, optional
            Multi-step forecasting strategy. Defaults to ``recursive``.
        kmeans : int, optional
            Number of clusters to use for centroid-based nearest neighbors.
        random_state : int, optional
            Seed for reproducible clustering.
        nan_strategy : {"propagate", "interpolate", "drop"}, optional
            How to treat NaN values in ``X``. ``propagate`` keeps them as-is,
            ``interpolate`` fills missing entries using linear interpolation and
            ``drop`` removes them. Defaults to ``propagate``.
        weight_by : {"recency", "distance"}, optional
            Strategy used when ``cf`` is ``"weighted"``. ``"recency"``
            weights neighbors by how recent they are, while ``"distance"``
            uses the actual distance value. Defaults to ``"recency"``.
        """
        self.config = TSKNNConfig(
            k=k,
            cf=cf,
            transform=transform,
            lags=lags,
            distance=distance,
            h=h,
            msas=msas,
            kmeans=kmeans,
            random_state=random_state,
            nan_strategy=nan_strategy,
            weight_by=weight_by,
        )

        if isinstance(self.config.k, str):
            self.k_strategy = self.config.k
            self.k_list = None
        elif isinstance(self.config.k, (list, tuple)):
            self.k_strategy = "combine"
            self.k_list = list(self.config.k)
        else:
            self.k_strategy = "single"
            self.k = int(self.config.k)
            self.k_list = None

        self.cf = self.config.cf.lower()
        self.transform = (
            self.config.transform.lower() if self.config.transform else None
        )
        if isinstance(self.config.lags, int):
            self.lags = np.arange(1, self.config.lags + 1)
        else:
            self.lags = np.array(list(self.config.lags))

        self.max_lag = int(np.max(self.lags))
        self.func_distance = get_distance(self.config.distance)
        self.h = self.config.h
        self.msas = self.config.msas.lower()
        self.h_ef = 1 if self.msas == "recursive" else self.h
        self.kmeans = self.config.kmeans
        self.random_state = self.config.random_state
        self.nan_strategy = self.config.nan_strategy.lower()
        self.weight_by = self.config.weight_by.lower()

        self._validate_params()

    def _validate_params(self) -> None:
        """Validate configuration options."""
        if self.weight_by not in {"recency", "distance"}:
            raise ValueError("weight_by must be 'recency' or 'distance'")
        if self.msas not in {"recursive", "mimo", "direct"}:
            raise ValueError("msas must be 'recursive', 'mimo' or 'direct'")
        if self.cf not in {"mean", "median", "weighted", "trimmed"}:
            raise ValueError("Invalid combination function")
        if self.nan_strategy not in {"propagate", "interpolate", "drop"}:
            raise ValueError("Unknown nan_strategy")

    def _handle_missing(self, x: np.ndarray) -> np.ndarray:
        """Return ``x`` after applying the configured NaN strategy."""
        if self.nan_strategy == "propagate":
            return x
        if self.nan_strategy == "drop":
            return x[~np.isnan(x)]
        if self.nan_strategy == "interpolate":
            nans = np.isnan(x)
            if nans.any():
                not_nans = np.where(~nans)[0]
                if not_nans.size:
                    x[nans] = np.interp(np.flatnonzero(nans), not_nans, x[not_nans])
            return x
        raise ValueError("Unknown nan_strategy")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "tsknn":
        """Fit the model using the provided time series ``X``."""
        if self.k_strategy == "sqrt":
            self.k = max(1, int(np.sqrt(len(X))))
        if self.msas == "mimo" and (
            self.h + self.max_lag + (self.k if hasattr(self, "k") else max(self.k_list))
            >= X.shape[0]
        ):
            raise ValueError(
                "You need a bigger series, or change the mode to recursive"
            )
        X = np.asarray(X, dtype=float)
        self.X = self._handle_missing(X)

        self.windowed_arr = sliding_window_view(
            self.X[:-1], window_shape=(self.max_lag,), axis=0
        )
        self.windowed_arr = self.windowed_arr[:, self.max_lag - self.lags[::-1]]

        if self.transform == "multiplicative" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr / self.x_mean[:, np.newaxis]
        elif self.transform == "additive" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr - self.x_mean[:, np.newaxis]

        self.windowed_arr = self.windowed_arr[
            : (1 - self.h_ef) if 1 - self.h_ef != 0 else None, :
        ]

        if self.kmeans:
            from sklearn.cluster import KMeans

            kmeans_model = KMeans(
                n_clusters=self.kmeans, random_state=self.random_state
            )
            kmeans_model.fit(self.windowed_arr)
            self.kmeans_means = np.array(
                [
                    np.mean(self.windowed_arr[kmeans_model.labels_ == i], axis=0)
                    for i in range(self.kmeans)
                ]
            )
            self.kmeans_labels = kmeans_model.labels_

            if self.transform == "multiplicative":
                self.x_mean = self.kmeans_means.mean(axis=1)
                self.kmeans_means = self.kmeans_means / self.x_mean[:, np.newaxis]
            elif self.transform == "additive":
                self.x_mean = self.kmeans_means.mean(axis=1)
                self.kmeans_means = self.kmeans_means - self.x_mean[:, np.newaxis]

        return self

    def _prepare_pred(self, x_pred: np.ndarray) -> np.ndarray:
        """Return ``x_pred`` after applying transformation and store its mean."""
        if self.transform == "multiplicative":
            self.x_pred_mean = x_pred.mean()
            return x_pred / self.x_pred_mean
        if self.transform == "additive":
            self.x_pred_mean = x_pred.mean()
            return x_pred - self.x_pred_mean
        return x_pred

    def _distance_array(self, x_pred: np.ndarray, offset: int) -> np.ndarray:
        """Compute distances from ``x_pred`` to training windows."""
        if self.kmeans:
            return self.func_distance(self.kmeans_means, x_pred)
        arr = (
            self.windowed_arr[: len(self.windowed_arr) - offset]
            if offset
            else self.windowed_arr
        )
        return self.func_distance(arr, x_pred)

    def _get_k_closest_positions(
        self, x_pred: np.ndarray, k: Optional[int] = None, offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of the ``k`` nearest neighbors for ``x_pred``."""
        x_pred = self._prepare_pred(x_pred)
        rolled_result = self._distance_array(x_pred, offset)
        k_val = k if k is not None else self.k
        index_closests = np.argpartition(rolled_result, range(k_val))[:k_val]
        if self.weight_by == "distance":
            distances = rolled_result[index_closests]
        else:
            distances = self.X.shape[0] - index_closests
        return index_closests, distances

    def _get_k_closest(self, k_closest: np.ndarray, offset: int = 0) -> np.ndarray:
        """Return the sequences corresponding to the provided neighbor indices."""
        if self.kmeans:
            resultado_final = np.zeros((len(k_closest), self.h_ef))
            for j, cluster in enumerate(k_closest):
                eqcluster = [
                    index
                    for index, value in enumerate(self.kmeans_labels == cluster)
                    if value
                ]

                resultado = np.zeros((len(eqcluster), self.h_ef))
                for i, pos in enumerate(eqcluster):
                    if pos + self.h_ef <= self.X.shape[0]:
                        resultado[i] = self.X[pos : pos + self.h_ef]
                resultado = resultado.mean(0)
                resultado_final[j] = resultado

            if self.transform == "multiplicative":
                return (
                    resultado_final / self.x_mean[k_closest, np.newaxis]
                ) * self.x_pred_mean
            elif self.transform == "additive":
                return (
                    resultado_final - self.x_mean[k_closest, np.newaxis]
                ) + self.x_pred_mean

            return resultado_final

        k_closest = (
            k_closest[:, np.newaxis]
            + np.tile(np.arange(self.h_ef), (len(k_closest), 1))
            + self.max_lag
            + offset
        )

        if self.transform == "multiplicative":
            X = self.X[self.max_lag :]
            return (
                np.take(X, k_closest - self.max_lag)
                / (self.x_mean[k_closest[:, 0] - self.max_lag, np.newaxis])
            ) * self.x_pred_mean
        if self.transform == "additive":
            X = self.X[self.max_lag :]
            return (
                np.take(X, k_closest - self.max_lag)
                - (self.x_mean[k_closest[:, 0] - self.max_lag, np.newaxis])
            ) + self.x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(
        self, k_closest: np.ndarray, distances: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Aggregate neighbor sequences according to the chosen strategy."""
        if self.cf == "mean":
            return k_closest.mean(axis=0)
        elif self.cf == "median":
            return np.median(k_closest, axis=0)
        elif self.cf == "weighted":
            eps = 1e-8
            reciprocal_d = 1 / np.sqrt(distances + eps)
            return np.average(k_closest, axis=0, weights=reciprocal_d)
        elif self.cf == "trimmed":
            if k_closest.shape[0] <= 2:
                return k_closest.mean(axis=0)
            trimmed = np.sort(k_closest, axis=0)[1:-1]
            return trimmed.mean(axis=0)
        return k_closest.mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return forecasts for the next ``h`` steps using context ``X``."""

        if X.shape[0] != self.max_lag:
            raise ValueError(
                "The biggest lag is different of the  length of example to predict"
            )

        def _predict_internal(k_val):
            if self.msas == "recursive":
                y_preds = []
                x_curr = X.copy()
                for _ in range(self.h):
                    idx, dist = self._get_k_closest_positions(x_curr, k=k_val)
                    k_close = self._get_k_closest(idx)
                    y_pred = self._get_mean(k_close, dist)[0]
                    y_preds.append(y_pred)
                    x_curr = np.concatenate((x_curr, np.array([y_pred])), axis=0)[
                        -self.max_lag :
                    ]
                return np.array(y_preds)
            elif self.msas == "mimo":
                idx, dist = self._get_k_closest_positions(X, k=k_val)
                k_close = self._get_k_closest(idx)
                return self._get_mean(k_close, dist)
            elif self.msas == "direct":
                y_preds = []
                for j in range(1, self.h + 1):
                    idx, dist = self._get_k_closest_positions(X, k=k_val, offset=j - 1)
                    k_close = self._get_k_closest(idx, offset=j - 1)
                    y_pred = self._get_mean(k_close, dist)[0]
                    y_preds.append(y_pred)
                return np.array(y_preds)

        if self.k_strategy == "combine":
            results = [_predict_internal(kv) for kv in self.k_list]
            return np.mean(results, axis=0)
        else:
            k_val = self.k if hasattr(self, "k") else max(self.k_list)
            return _predict_internal(k_val)

    def save(self, path: str) -> None:
        """Serialize model to ``path`` using :mod:`pickle`."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "tsknn":
        """Load a model instance from ``path``."""
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a tsknn instance")
        return obj


class mtsknn(BaseEstimator, RegressorMixin):
    """Multivariate wrapper around :class:`tsknn`.

    This class fits one ``tsknn`` model per column of a multivariate series
    and returns joint forecasts for all of them.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models: List[tsknn] = []
        self.n_features = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "mtsknn":
        """Fit one ``tsknn`` model per variable in ``X``."""

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.n_features = X.shape[1]
        self.models = [tsknn(**self.kwargs) for _ in range(self.n_features)]
        for i, model in enumerate(self.models):
            model.fit(X[:, i])
        self.h = self.models[0].h
        self.max_lag = self.models[0].max_lag
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return forecasts for all variables in ``X``."""

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[1] != self.n_features:
            raise ValueError("Number of series in X does not match fitted model")
        preds = [model.predict(X[:, i]) for i, model in enumerate(self.models)]
        return np.column_stack(preds)

    def save(self, path: str) -> None:
        """Serialize multivariate model to ``path`` using :mod:`pickle`."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "mtsknn":
        """Load a :class:`mtsknn` instance from ``path``."""
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a mtsknn instance")
        return obj
