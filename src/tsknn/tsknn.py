"""Core KNN utilities for time series forecasting."""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
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


def get_distance(distance: str = "euclidean") -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a distance function identified by ``distance``."""

    if distance == "euclidean":
        return sum_euclidean
    if distance == "manhattan":
        return sum_manhattan
    if distance == "chebyshev":
        return max_chebyshev
    if distance == "cosine":
        return cosine_distance
    return sum_euclidean


def select_lags_pacf(x: Sequence[float], nlags: int, threshold: float = 0.2) -> List[int]:
    """Return lag indices with partial autocorrelation above ``threshold``."""

    pacf_vals = pacf(x, nlags=nlags)
    lags = [i for i, val in enumerate(pacf_vals[1:], start=1) if abs(val) >= threshold]
    return lags if lags else list(range(1, nlags + 1))


class tsknn:
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
        """
        if isinstance(k, str):
            self.k_strategy = k
            self.k_list = None
        elif isinstance(k, (list, tuple)):
            self.k_strategy = "combine"
            self.k_list = list(k)
        else:
            self.k_strategy = "single"
            self.k = int(k)
            self.k_list = None
        self.cf = cf.lower()
        self.transform = transform.lower() if transform else None
        if isinstance(lags, int):
            self.lags = np.arange(1, lags + 1)
        else:
            self.lags = np.array(list(lags))
        self.max_lag = int(np.max(self.lags))
        self.func_distance = get_distance(distance)
        self.h = h
        self.msas = msas.lower()
        if self.msas == "recursive":
            self.h_ef = 1
        else:
            self.h_ef = h
        self.kmeans = kmeans
        self.random_state = random_state


    def fit(self, X: np.ndarray) -> None:
        """Fit the model using the provided time series ``X``."""
        if self.k_strategy == "sqrt":
            self.k = max(1, int(np.sqrt(len(X))))
        if self.msas == 'mimo' and (self.h + self.max_lag + (self.k if hasattr(self, 'k') else max(self.k_list)) >= X.shape[0]):
            raise ValueError('You need a bigger series, or change the mode to recursive')
        self.X = X

        self.windowed_arr = sliding_window_view(self.X[:-1], window_shape=(self.max_lag,), axis=0)
        self.windowed_arr = self.windowed_arr[:, self.max_lag - self.lags[::-1]]

        if self.transform == "multiplicative" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr / self.x_mean[:, np.newaxis]
        elif self.transform == "additive" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr - self.x_mean[:, np.newaxis]

        self.windowed_arr = self.windowed_arr[:(1 - self.h_ef) if 1 - self.h_ef != 0 else None, :]

        if self.kmeans:
            from sklearn.cluster import KMeans
            kmeans_model = KMeans(n_clusters=self.kmeans, random_state=self.random_state)
            kmeans_model.fit(self.windowed_arr)
            self.kmeans_means = np.array([np.mean(self.windowed_arr[kmeans_model.labels_ == i], axis=0) 
                                    for i in range(self.kmeans)])
            self.kmeans_labels = kmeans_model.labels_

            if self.transform == "multiplicative":
                self.x_mean = self.kmeans_means.mean(axis=1)
                self.kmeans_means = self.kmeans_means / self.x_mean[:, np.newaxis]
            elif self.transform == "additive":
                self.x_mean = self.kmeans_means.mean(axis=1)
                self.kmeans_means = self.kmeans_means - self.x_mean[:, np.newaxis]


    def _get_k_closest_positions(
        self, x_pred: np.ndarray, k: Optional[int] = None, offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of the ``k`` nearest neighbors for ``x_pred``."""

        if self.transform == "multiplicative":
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred / self.x_pred_mean
        elif self.transform == "additive":
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred - self.x_pred_mean

        # rolled_result = np.apply_along_axis(lambda x: self.func_distance(x, x_pred), axis=-1, arr=self.windowed_arr)
        # rolled_result = np.sum((self.windowed_arr - x_pred)**2, axis=-1)
        if self.kmeans:
            rolled_result = self.func_distance(self.kmeans_means, x_pred)
        else:
            arr = self.windowed_arr[:len(self.windowed_arr)-offset] if offset else self.windowed_arr
            rolled_result = self.func_distance(arr, x_pred)
        k_val = k if k is not None else self.k
        index_closests = np.argpartition(rolled_result, range(k_val))[:k_val]
        distances = self.X.shape[0] - index_closests
        return index_closests, distances

    def _get_k_closest(self, k_closest: np.ndarray, offset: int = 0) -> np.ndarray:
        """Return the sequences corresponding to the provided neighbor indices."""
        if self.kmeans:
            resultado_final = np.zeros((len(k_closest), self.h_ef))
            for j, cluster in enumerate(k_closest):
                eqcluster = [index for index, value in enumerate(self.kmeans_labels == cluster) if value]

                resultado = np.zeros((len(eqcluster), self.h_ef))
                for i, pos in enumerate(eqcluster):
                    if pos + self.h_ef <= self.X.shape[0]:
                        resultado[i] = self.X[pos:pos + self.h_ef]
                resultado = resultado.mean(0)
                resultado_final[j] = resultado

            if self.transform == "multiplicative":
                return (resultado_final / self.x_mean[k_closest, np.newaxis]) * self.x_pred_mean
            elif self.transform == "additive":
                return (resultado_final - self.x_mean[k_closest, np.newaxis]) + self.x_pred_mean

            return resultado_final

        k_closest = k_closest[:, np.newaxis] + np.tile(np.arange(self.h_ef), (len(k_closest), 1)) + self.max_lag + offset

        if self.transform == "multiplicative":
            X = self.X[self.max_lag:]
            return (np.take(X, k_closest - self.max_lag) / (self.x_mean[k_closest[:, 0] - self.max_lag, np.newaxis])) * self.x_pred_mean
        elif self.transform == "additive":
            X = self.X[self.max_lag:]
            return (np.take(X, k_closest - self.max_lag) - (self.x_mean[k_closest[:, 0] - self.max_lag, np.newaxis])) + self.x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(
        self, k_closest: np.ndarray, distances: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Aggregate neighbor sequences according to the chosen strategy."""
        if self.cf == "mean":
            return k_closest.mean(axis=0)
        elif self.cf == "median":
            return np.median(k_closest, axis=0)
        elif self.cf == "weighted":  # to do, fix para o caso mimo
            reciprocal_d = 1 / np.sqrt(distances)
            return reciprocal_d.dot(k_closest)[0] / reciprocal_d.sum()
        elif self.cf == "trimmed":
            if k_closest.shape[0] <= 2:
                return k_closest.mean(axis=0)
            trimmed = np.sort(k_closest, axis=0)[1:-1]
            return trimmed.mean(axis=0)
        return k_closest.mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return forecasts for the next ``h`` steps using context ``X``."""

        if X.shape[0] != self.max_lag:
            raise ValueError('The biggest lag is different of the  length of example to predict')

        def _predict_internal(k_val):
            if self.msas == "recursive":
                y_preds = []
                x_curr = X.copy()
                for _ in range(self.h):
                    idx, dist = self._get_k_closest_positions(x_curr, k=k_val)
                    k_close = self._get_k_closest(idx)
                    y_pred = self._get_mean(k_close, dist)[0]
                    y_preds.append(y_pred)
                    x_curr = np.concatenate((x_curr, np.array([y_pred])), axis=0)[-self.max_lag:]
                return np.array(y_preds)
            elif self.msas == "mimo":
                idx, dist = self._get_k_closest_positions(X, k=k_val)
                k_close = self._get_k_closest(idx)
                return self._get_mean(k_close, dist)
            elif self.msas == "direct":
                y_preds = []
                for j in range(1, self.h + 1):
                    idx, dist = self._get_k_closest_positions(X, k=k_val, offset=j-1)
                    k_close = self._get_k_closest(idx, offset=j-1)
                    y_pred = self._get_mean(k_close, dist)[0]
                    y_preds.append(y_pred)
                return np.array(y_preds)

        if self.k_strategy == "combine":
            results = [_predict_internal(kv) for kv in self.k_list]
            return np.mean(results, axis=0)
        else:
            k_val = self.k if hasattr(self, "k") else max(self.k_list)
            return _predict_internal(k_val)
