from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import scipy
np.set_printoptions(suppress=True)


def get_distance(distance="euclidean"):
    if distance == "euclidean":
        return scipy.spatial.distance.sqeuclidean
    if distance == "minkowski":
        return scipy.spatial.distance.minkowski
    if distance == "cityblock":
        return scipy.spatial.distance.cityblock
    if distance == "canberra":
        return scipy.spatial.distance.canberra
    if distance == "mahalanobis":
        return scipy.spatial.distance.mahalanobis
    return scipy.spatial.distance.sqeuclidean


class tsknn:
    def __init__(self,
                 k=3,
                 cf="mean",
                 transform=None,  # "additive", "multiplicative"
                 lags=3,
                 distance="euclidean",
                 h=12,
                 msas="recursive"
                 ):
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

    def fit(self, X):
        self.X = X

    def _get_k_closest_positions(self, x_pred):
        '''
        Return the position of the k nearest neighbors, the first is the closest
        '''
        windowed_arr = sliding_window_view(self.X[:-1], window_shape=(self.lags,), axis=0)

        if self.transform == "multiplicative":
            self.x_mean = windowed_arr.mean(axis=1)
            windowed_arr = windowed_arr.copy()/self.x_mean[:, np.newaxis]
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred / self.x_pred_mean
        elif self.transform == "additive":
            self.x_mean = windowed_arr.mean(axis=1)
            windowed_arr = windowed_arr.copy() - self.x_mean[:, np.newaxis]
            self.x_pred_mean = x_pred.mean()
            x_pred = x_pred - self.x_pred_mean

        windowed_arr = windowed_arr[:(1-self.h_ef) if 1-self.h_ef != 0 else None,:]
        rolled_result = np.apply_along_axis(lambda x: self.func_distance(x, x_pred), axis=-1, arr=windowed_arr)
        index_closests = np.argpartition(rolled_result, range(self.k))[:self.k]
        distances = self.X.shape[0] - index_closests
        return index_closests, distances

    def _get_k_closest(self, k_closest):
        '''
        Return the sequences to the knns
        '''
        k_closest = k_closest[:, np.newaxis] + np.tile(np.arange(self.h_ef), (len(k_closest), 1)) + self.k

        if self.transform == "multiplicative":
            X = self.X[self.k:] #/ self.x_mean
            #return np.take(X, k_closest-s;elf.k)/(self.x_mean[k_closest-self.k]) * self.x_pred_mean
            return (np.take(X, k_closest-self.k)/(self.x_mean[k_closest[:,0]-self.k,np.newaxis])) * self.x_pred_mean
        elif self.transform == "additive":
            X = self.X[self.k:] - self.x_mean
            return np.take(X, k_closest-self.k) + self.x_pred_mean

        return np.take(self.X, k_closest)

    def _get_mean(self, k_closest, distances=None):
        '''
        Return the mean selected by the user
        '''
        if self.cf == "mean":
            return k_closest.mean(axis=0)
        elif self.cf == "median":
            return np.median(k_closest, axis=0)
        elif self.cf == "weighted":  # to do, fix para o caso mimo
            reciprocal_d = 1 / np.sqrt(distances)
            return reciprocal_d.dot(k_closest)[0]/reciprocal_d.sum()
        return k_closest.mean()

    def predict(self, X):
        if self.msas == "recursive":
            y_preds = []
            for i in range(self.h):
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
