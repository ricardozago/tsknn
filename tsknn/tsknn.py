import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
        random_state (int|None): Seed for reproducibility if in the future we add some randomness.
    """

    def __init__(
        self,
        k=3,
        cf: str = "mean",
        transform: str = None,
        lags=3,
        distance: str = "euclidean",
        h: int = 12,
        msas: str = "recursive",
        random_state: int = 42,
        force_stable: bool = False,
    ):
        if not (isinstance(k, int) and k >= 1) and not (
            isinstance(k, list) and all(isinstance(ki, int) and ki >= 1 for ki in k)
        ):
            raise ValueError(
                "k must be a positive integer or a list of positive integers."
            )
        if cf.lower() not in {"mean", "median", "weighted"}:
            raise ValueError("cf must be 'mean', 'median' or 'weighted'.")
        if transform is not None and transform.lower() not in {
            "additive",
            "multiplicative",
        }:
            raise ValueError(
                f"transform must be 'additive', 'multiplicative' or None, not {transform}"
            )
        if not (
            isinstance(lags, int)
            or (isinstance(lags, list) and all(isinstance(lag, int) for lag in lags))
        ):
            raise ValueError("lags must be an integer or a list of integers.")
        if not isinstance(h, int) or h < 1:
            raise ValueError("h must be a positive integer.")
        if msas.lower() not in {"recursive", "mimo"}:
            raise ValueError("msas must be 'recursive' or 'mimo'.")
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
        self.h_ef = 1 if self.msas == "recursive" else h
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
        if self.msas == "mimo" and (
            self.h + self.maxlags + np.max(self.k) >= X.shape[0]
        ):
            raise ValueError(
                "You need a bigger series, or change the mode to recursive"
            )
        self.X = X

        self.windowed_arr = sliding_window_view(
            self.X[:-1], window_shape=(self.maxlags,), axis=0
        )

        if isinstance(self.lags, list):
            self.lag_indices = np.array([self.maxlags - lag for lag in self.lags])
            self.windowed_arr = self.windowed_arr[:, self.lag_indices]
        else:
            self.lag_indices = None

        if self.transform == "multiplicative":
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr / self.x_mean[:, np.newaxis]
        elif self.transform == "additive":
            self.x_mean = self.windowed_arr.mean(axis=1)
            self.windowed_arr = self.windowed_arr - self.x_mean[:, np.newaxis]

        self.windowed_arr = self.windowed_arr[
            : (1 - self.h_ef) if 1 - self.h_ef != 0 else None, :
        ]

    def _get_k_closest_positions(
        self, x_pred: np.ndarray, k: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the positions of the k nearest neighbors.
        Args:
            x_pred (np.ndarray): Prediction vector (n_lags,).
        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices of neighbors and distances.
        """

        if self.lag_indices is not None:
            x_pred = x_pred[self.lag_indices]

        if self.transform == "multiplicative":
            x_pred_mean = x_pred.mean()
            x_pred = x_pred / x_pred_mean
        elif self.transform == "additive":
            x_pred_mean = x_pred.mean()
            x_pred = x_pred - x_pred_mean
        else:
            x_pred_mean = None

        # Unnoptimized distance calculation
        # rolled_distances = np.apply_along_axis(lambda x: self.func_distance(x, x_pred), axis=-1, arr=self.windowed_arr)
        # rolled_distances = np.sum((self.windowed_arr - x_pred)**2, axis=-1)

        rolled_distances = self.func_distance(self.windowed_arr, x_pred)
        index_closests = np.argpartition(rolled_distances, range(k))[:k]
        if self.force_stable:
            index_closests = np.argsort(rolled_distances, stable=True)[:k]
        return index_closests, rolled_distances, x_pred_mean

    def _get_k_closest(
        self, k_closest: np.ndarray, x_pred_mean: np.ndarray
    ) -> np.ndarray:
        """
        Returns the sequences of the k nearest neighbors.
        Args:
            k_closest (np.ndarray): Indices of the nearest neighbors.
        Returns:
            np.ndarray: Sequences of the neighbors.
        """

        k_closest = (
            k_closest[:, np.newaxis]
            + np.tile(np.arange(self.h_ef), (len(k_closest), 1))
            + self.maxlags
        )

        if self.transform == "multiplicative":
            X = self.X[self.maxlags :]
            return (
                np.take(X, k_closest - self.maxlags)
                / (self.x_mean[k_closest[:, 0] - self.maxlags, np.newaxis])
            ) * x_pred_mean
        elif self.transform == "additive":
            X = self.X[self.maxlags :]
            return (
                np.take(X, k_closest - self.maxlags)
                - (self.x_mean[k_closest[:, 0] - self.maxlags, np.newaxis])
            ) + x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(
        self,
        k_closest: np.ndarray,
        index_closests: np.ndarray,
        rolled_distances: np.ndarray,
    ) -> np.ndarray:
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
        elif self.cf == "weighted":
            d = rolled_distances[index_closests]
            if np.abs(d[0]) < 1e-14:
                return k_closest[0]
            reciprocal_d = 1 / np.sqrt(d)
            return reciprocal_d.dot(k_closest) / reciprocal_d.sum()
        return k_closest.mean(axis=0)

    def predict(self, X: np.ndarray = None) -> np.ndarray:
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
            X = self.X[-self.maxlags :]
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a np.ndarray.")
        if X.ndim != 1:
            raise ValueError("X must be a one-dimensional array.")
        if X.shape[0] != self.maxlags:
            raise ValueError(
                "The biggest lag is different from the length of the example to predict"
            )
        k_list = self.k if isinstance(self.k, list) else [self.k]
        results = []
        for k in k_list:
            if self.msas == "recursive":
                y_preds = []
                X_temp = np.copy(X)
                for _ in range(self.h):
                    index_closests, rolled_distances, x_pred_mean = (
                        self._get_k_closest_positions(X_temp, k)
                    )
                    k_closest = self._get_k_closest(index_closests, x_pred_mean)
                    y_pred = self._get_mean(
                        k_closest, index_closests, rolled_distances
                    )[0]
                    y_preds.append(y_pred)
                    X_temp = np.concatenate((X_temp, np.array([y_pred])), axis=0)[
                        -self.maxlags :
                    ]
                y_preds = np.array(y_preds)
            elif self.msas == "mimo":
                index_closests, rolled_distances, x_pred_mean = (
                    self._get_k_closest_positions(X, k)
                )
                k_closest = self._get_k_closest(index_closests, x_pred_mean)
                y_preds = self._get_mean(k_closest, index_closests, rolled_distances)
            results.append(y_preds)
        if len(results) == 1:
            return results[0]
        return np.mean(results, axis=0)


def sum_euclidean(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculates the sum of squared Euclidean distances between each row of M and vector v.
    Args:
        M (np.ndarray): Sample matrix (n_samples, n_features).
        v (np.ndarray): Comparison vector (n_features,).
    Returns:
        np.ndarray: Array of distances for each row of M.
    """
    # https://stackoverflow.com/a/49633639
    tmp = M - v
    return np.einsum("ij,ij->i", tmp, tmp)


def get_distance(distance: str = "euclidean"):
    """
    Returns the appropriate distance function.
    Args:
        distance (str): Name of the distance metric.
    Returns:
        Callable: Distance function.
    """
    if distance == "euclidean":
        return sum_euclidean
    return sum_euclidean
