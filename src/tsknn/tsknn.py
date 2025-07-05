"""Core KNN model implementation for time series forecasting."""

from typing import Optional, Sequence, Union

from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

np.set_printoptions(suppress=True)


def sum_euclidean(M, v):
    """Return the euclidean distance between ``v`` and each row of ``M``."""

    # https://stackoverflow.com/a/49633639
    tmp = M - v
    return np.einsum("ij,ij->i", tmp, tmp)


def get_distance(distance: str = "euclidean"):
    """Map a distance name to a distance function."""

    if distance == "euclidean":
        return sum_euclidean
    return sum_euclidean


class tsknn:
    """K-Nearest Neighbours regressor for univariate time series."""
    def __init__(
        self,
        k: int = 3,
        cf: str = "mean",
        transform: Optional[str] = None,
        lags: Union[int, Sequence[int]] = 3,
        distance: str = "euclidean",
        h: int = 12,
        msas: str = "recursive",
        kmeans: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Create a new :class:`tsknn` instance.

        Parameters
        ----------
        k : int
            Number of nearest neighbours to use.
        cf : {"mean", "median", "weighted"}
            Combination function used when aggregating neighbours.
        transform : {"additive", "multiplicative"}, optional
            Transformation applied to the windows before computing distances.
        lags : int or sequence of int
            Number of lagged observations in each sample window.
        distance : str
            Distance metric to use. Only ``"euclidean"`` is implemented.
        h : int
            Forecast horizon.
        msas : {"recursive", "mimo"}
            Multi-step strategy.
        kmeans : int, optional
            Number of clusters used to pre-cluster the windows.
        random_state : int, optional
            Random state passed to the ``KMeans`` constructor.
        """
        self.k = k
        self.cf = cf.lower()
        self.transform = transform.lower() if transform else None
        self.lags = lags
        self.func_distance = get_distance(distance)
        self.h = h
        self.msas = msas.lower()
        if self.msas == "recursive":
            self.h_ef = 1
        else:
            self.h_ef = h
        self.kmeans = kmeans
        self.random_state = random_state


    def fit(self, X: np.ndarray):
        """Fit the model using the provided time series ``X``."""

        if self.msas == "mimo" and (self.h + np.max(self.lags) + self.k >= X.shape[0]):
            raise ValueError("You need a bigger series, or change the mode to recursive")

        self.X = X

        self.windowed_arr = sliding_window_view(self.X[:-1], window_shape=(self.lags,), axis=0)

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


    def _get_k_closest_positions(self, x_pred):
        """Return indices of the ``k`` closest windows to ``x_pred``."""

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
            rolled_result = self.func_distance(self.windowed_arr, x_pred)
        index_closests = np.argpartition(rolled_result, range(self.k))[:self.k]
        distances = self.X.shape[0] - index_closests
        return index_closests, distances

    def _get_k_closest(self, k_closest):
        """Return the ``k`` closest sequences from ``k_closest`` indices."""
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

        k_closest = k_closest[:, np.newaxis] + np.tile(np.arange(self.h_ef), (len(k_closest), 1)) + self.lags

        if self.transform == "multiplicative":
            X = self.X[self.lags:]
            return (np.take(X, k_closest - self.lags) / (self.x_mean[k_closest[:, 0] - self.lags, np.newaxis])) * self.x_pred_mean
        elif self.transform == "additive":
            X = self.X[self.lags:]
            return (np.take(X, k_closest - self.lags) - (self.x_mean[k_closest[:, 0] - self.lags, np.newaxis])) + self.x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(self, k_closest, distances=None):
        """Aggregate neighbours according to ``cf`` setting."""
        if self.cf == "mean":
            return k_closest.mean(axis=0)
        elif self.cf == "median":
            return np.median(k_closest, axis=0)
        elif self.cf == "weighted":  # to do, fix para o caso mimo
            reciprocal_d = 1 / np.sqrt(distances)
            return reciprocal_d.dot(k_closest)[0] / reciprocal_d.sum()
        return k_closest.mean()

    def predict(self, X: np.ndarray):
        """Predict the next ``h`` values given the last ``lags`` samples."""

        if X.shape[0] != np.max(self.lags):
            raise ValueError(
                "The biggest lag is different of the  length of example to predict"
            )
        if self.msas == "recursive":
            y_preds = []
            for _ in range(self.h):
                index_closests, distances = self._get_k_closest_positions(X)
                k_closest = self._get_k_closest(index_closests)
                y_pred = self._get_mean(k_closest, distances)[0]
                y_preds.append(y_pred)
                X = np.concatenate((X, np.array([y_pred])), axis=0)[-self.lags:]
            y_preds = np.array(y_preds)
        elif self.msas == "mimo":
            index_closests, distances = self._get_k_closest_positions(X)
            k_closest = self._get_k_closest(index_closests)
            y_preds = self._get_mean(k_closest, distances)
        return y_preds
