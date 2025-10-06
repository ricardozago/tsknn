from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from .utils.distance import get_distance
np.set_printoptions(suppress=True)


class tsknn:
    """
    Time Series K-Nearest Neighbors (TSKNN) for time series forecasting.

    Args:
        k (int): Number of nearest neighbors.
        cf (str): Combination function ('mean', 'median', 'weighted').
        transform (str|None): Transformation to apply ('additive', 'multiplicative' or None).
        lags (int|list): Lags to be used.
        distance (str): Distance metric ('euclidean').
        h (int): Forecast horizon.
        msas (str): Multi-step strategy ('recursive' or 'mimo').
        kmeans (int|None): Number of clusters for KMeans (optional).
        random_state (int|None): Seed for reproducibility.
    """
    def __init__(self,
                 k: int = 3,
                 cf: str = "mean",
                 transform: str = None,
                 lags=3,
                 distance: str = "euclidean",
                 h: int = 12,
                 msas: str = "recursive",
                 kmeans: int = None,
                 random_state: int = None,
                 force_stable: bool = False
                 ):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if cf.lower() not in {"mean", "median", "weighted"}:
            raise ValueError("cf must be 'mean', 'median' or 'weighted'.")
        if transform is not None and transform.lower() not in {"additive", "multiplicative"}:
            raise ValueError(f"transform must be 'additive', 'multiplicative' or None, not {transform}")
        if not (isinstance(lags, int) or (isinstance(lags, list) and all(isinstance(lag, int) for lag in lags))):
            raise ValueError("lags must be an integer or a list of integers.")
        if not isinstance(h, int) or h < 1:
            raise ValueError("h must be a positive integer.")
        if msas.lower() not in {"recursive", "mimo"}:
            raise ValueError("msas must be 'recursive' or 'mimo'.")
        if kmeans is not None and (not isinstance(kmeans, int) or kmeans < 1):
            raise ValueError("kmeans must be None or a positive integer.")
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state must be None or an integer.")

        self.k = k
        self.cf = cf.lower()
        self.transform = transform.lower() if transform else None
        self.lags = lags
        self.maxlags = np.max(self.lags)
        self.func_distance = get_distance(distance)
        self.h = h
        self.msas = msas.lower()
        if self.msas == "recursive":
            self.h_ef = 1
        else:
            self.h_ef = h
        self.kmeans = kmeans
        self.random_state = random_state
        self.force_stable = force_stable

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the TSKNN model to the input data.
        Args:
            X (np.ndarray): Univariate time series (n_observations,).
        Raises:
            ValueError: If the series is too small for MIMO mode.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a np.ndarray.")
        if X.ndim != 1:
            raise ValueError("X must be a one-dimensional array.")
        if self.msas == 'mimo' and (self.h + self.maxlags + self.k >= X.shape[0]):
            raise ValueError(
                'You need a bigger series, or change the mode to recursive'
            )
        self.X = X

        self.windowed_arr = sliding_window_view(
            self.X[:-1], window_shape=(self.maxlags,), axis=0
        )

        if isinstance(self.lags, list):
            self.windowed_arr = self.windowed_arr[:, [self.maxlags - lag for lag in self.lags]]

        if self.transform == "multiplicative" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr / self.x_mean[:, np.newaxis]
        elif self.transform == "additive" and not self.kmeans:
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr - self.x_mean[:, np.newaxis]

        self.windowed_arr = self.windowed_arr[
            :(1 - self.h_ef) if 1 - self.h_ef != 0 else None, :
        ]

        # if self.kmeans:
        #     from sklearn.cluster import KMeans
        #     kmeans_model = KMeans(
        #         n_clusters=self.kmeans, random_state=self.random_state
        #     )
        #     kmeans_model.fit(self.windowed_arr)
        #     self.kmeans_means = np.array([
        #         np.mean(self.windowed_arr[kmeans_model.labels_ == i], axis=0)
        #         for i in range(self.kmeans)
        #     ])
        #     self.kmeans_labels = kmeans_model.labels_

        #     if self.transform == "multiplicative":
        #         self.x_mean = self.kmeans_means.mean(axis=1)
        #         self.kmeans_means = self.kmeans_means / self.x_mean[:, np.newaxis]
        #     elif self.transform == "additive":
        #         self.x_mean = self.kmeans_means.mean(axis=1)
        #         self.kmeans_means = self.kmeans_means - self.x_mean[:, np.newaxis]

    def _get_k_closest_positions(self, x_pred: np.ndarray):
        """
        Returns the positions of the k nearest neighbors.
        Args:
            x_pred (np.ndarray): Prediction vector (n_lags,).
        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices of neighbors and distances.
        """

        # rolled_distances = np.apply_along_axis(lambda x: self.func_distance(x, x_pred), axis=-1, arr=self.windowed_arr)
        # rolled_distances = np.sum((self.windowed_arr - x_pred)**2, axis=-1)

        if isinstance(self.lags, list):
            x_pred = x_pred[[self.maxlags - x for x in self.lags]]

        if self.transform == "multiplicative":
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred / self.x_pred_mean
        elif self.transform == "additive":
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred - self.x_pred_mean

        if self.kmeans:
            rolled_distances = self.func_distance(self.kmeans_means, x_pred)
        else:
            rolled_distances = self.func_distance(self.windowed_arr, x_pred)
        index_closests = np.argpartition(rolled_distances, range(self.k))[:self.k]
        if self.force_stable:
            index_closests = np.argsort(rolled_distances, stable=True)[:self.k]
        # distances = self.X.shape[0] - index_closests
        return index_closests, rolled_distances

    def _get_k_closest(self, k_closest: np.ndarray) -> np.ndarray:
        """
        Returns the sequences of the k nearest neighbors.
        Args:
            k_closest (np.ndarray): Indices of the nearest neighbors.
        Returns:
            np.ndarray: Sequences of the neighbors.
        """
        # if self.kmeans:
        #     result_final = np.zeros((len(k_closest), self.h_ef))
        #     for j, cluster in enumerate(k_closest):
        #         eqcluster = [idx for idx, value in enumerate(self.kmeans_labels == cluster) if value]

        #         result = np.zeros((len(eqcluster), self.h_ef))
        #         for i, pos in enumerate(eqcluster):
        #             if pos + self.h_ef <= self.X.shape[0]:
        #                 result[i] = self.X[pos:pos + self.h_ef]
        #         result = result.mean(0)
        #         result_final[j] = result

        #     if self.transform == "multiplicative":
        #         return (result_final / self.x_mean[k_closest, np.newaxis]) * self.x_pred_mean
        #     elif self.transform == "additive":
        #         return (result_final - self.x_mean[k_closest, np.newaxis]) + self.x_pred_mean

        #     return result_final

        k_closest = (
            k_closest[:, np.newaxis]
            + np.tile(np.arange(self.h_ef), (len(k_closest), 1))
            + self.maxlags
        )

        if self.transform == "multiplicative":
            X = self.X[self.maxlags:]
            return (
                np.take(X, k_closest - self.maxlags)
                / (self.x_mean[k_closest[:, 0] - self.maxlags, np.newaxis])
            ) * self.x_pred_mean
        elif self.transform == "additive":
            X = self.X[self.maxlags:]
            return (
                np.take(X, k_closest - self.maxlags)
                - (self.x_mean[k_closest[:, 0] - self.maxlags, np.newaxis])
            ) + self.x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(self, 
                  k_closest: np.ndarray,
                  index_closests: np.ndarray,
                  rolled_distances: np.ndarray) -> np.ndarray:
        """
        Returns the mean of the neighbors according to the combination function.
        Args:
            k_closest (np.ndarray): Sequences of the neighbors.
            distances (np.ndarray, optional): Distances for weighting.
        Returns:
            np.ndarray: Combined value of the neighbors.
        """
        if self.cf == "mean":
            return k_closest.mean(axis=0)
        elif self.cf == "median":
            return np.median(k_closest, axis=0)
        elif self.cf == "weighted":  # to do, fix for mimo case
            reciprocal_d = 1 / np.sqrt(rolled_distances[index_closests])
            return reciprocal_d.dot(k_closest) / reciprocal_d.sum()
        return k_closest.mean(axis=0)

    def predict(self, X=None) -> np.ndarray:
        """
        Makes predictions for the defined horizon.
        Args:
            X (np.ndarray): Input vector with size equal to the largest lag.
        Returns:
            np.ndarray: Predictions for horizon h.
        Raises:
            ValueError: If the size of X is not equal to the largest lag.
        """
        if X is None:
            X = self.X[-self.maxlags:]
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a np.ndarray.")
        if X.ndim != 1:
            raise ValueError("X must be a one-dimensional array.")
        if X.shape[0] != self.maxlags:
            raise ValueError('The biggest lag is different from the length of the example to predict')
        if self.msas == "recursive":
            y_preds = []
            for _ in range(self.h):
                index_closests, rolled_distances = self._get_k_closest_positions(X)
                k_closest = self._get_k_closest(index_closests)
                y_pred = self._get_mean(k_closest, index_closests, rolled_distances)[0]
                y_preds.append(y_pred)
                X = np.concatenate((X, np.array([y_pred])), axis=0)[-self.maxlags:]
            y_preds = np.array(y_preds)
        elif self.msas == "mimo":
            index_closests, rolled_distances = self._get_k_closest_positions(X)
            k_closest = self._get_k_closest(index_closests)
            y_preds = self._get_mean(k_closest, index_closests, rolled_distances)
        return y_preds
