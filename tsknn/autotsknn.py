import numpy as np
import optuna
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.stattools import adfuller, pacf

from tsknn.tsknn import tsknn


class autotsknn:
    def __init__(
        self,
        metric="mse",
        validation_splits=None,
        n_trials=100,
    ):
        if validation_splits is None:
            validation_splits = [0.2]
        self.metric = self._get_metric(metric)
        self.metric_name = metric
        self.validation_splits = validation_splits
        self.n_trials = n_trials

    def _get_metric(self, metric_name):
        if metric_name == "mse":
            return mean_squared_error
        elif metric_name == "mae":
            return mean_absolute_error
        elif metric_name == "mape":
            return mean_absolute_percentage_error
        else:
            raise ValueError(f"Metric {metric_name} not supported.")

    def _is_stationary(self, X):
        result = adfuller(X)
        return result[1] <= 0.05

    def _get_pacf_lags(self, X):
        pacf_values = pacf(X)
        significant_lags = [
            i for i, value in enumerate(pacf_values) if value > 1.96 / np.sqrt(len(X))
        ]
        if len(significant_lags) > 1:
            return significant_lags[1:]
        else:
            return [1]

    def _objective(self, trial, X, lags):
        errors = []
        for split in self.validation_splits:
            train_size = int(len(X) * (1 - split))
            train, test = X[0:train_size], X[train_size : len(X)]

            k_max = min(100, int(len(train) * 0.1))
            params = {
                "k": trial.suggest_int("k", 1, k_max),
                "cf": trial.suggest_categorical("cf", ["mean", "median", "weighted"]),
                "distance": trial.suggest_categorical(
                    "distance", ["euclidean", "manhattan"]
                ),
                "h": len(test),
            }

            if self._is_stationary(train):
                params["transform"] = None
            else:
                params["transform"] = trial.suggest_categorical(
                    "transform", ["additive", "multiplicative"]
                )

            if lags == "auto":
                n_lags = trial.suggest_int(
                    "n_lags", 1, min(20, len(train) - len(test) - 1)
                )
                params["lags"] = list(range(1, n_lags + 1))
            else:
                params["lags"] = lags

            model = tsknn(**params)
            model.fit(train)
            preds = model.predict()
            errors.append(self.metric(test, preds))

        return np.mean(errors)

    def fit(self, X):
        # Study 1: PACF-guided lags
        pacf_lags = self._get_pacf_lags(X)
        study_pacf = optuna.create_study(direction="minimize")
        study_pacf.optimize(
            lambda trial: self._objective(trial, X, pacf_lags), n_trials=self.n_trials
        )

        # Study 2: Full search for lags
        study_auto = optuna.create_study(direction="minimize")
        study_auto.optimize(
            lambda trial: self._objective(trial, X, "auto"), n_trials=self.n_trials
        )

        # Compare results and choose the best
        if study_pacf.best_value < study_auto.best_value:
            self.best_params_ = study_pacf.best_params
            self.best_value_ = study_pacf.best_value
            self.best_params_["lags"] = pacf_lags
            lags_source = "PACF"
        else:
            self.best_params_ = study_auto.best_params
            self.best_value_ = study_auto.best_value
            lags_source = "Auto Search"

        print("--- Best Hyperparameters ---")
        print(f"Lags Source: {lags_source}")
        for key, value in self.best_params_.items():
            print(f"{key}: {value}")
        print(f"Best Validation {self.metric_name.upper()}: {self.best_value_}")

        final_params = self.best_params_.copy()
        if "n_lags" in final_params:
            n_lags = final_params.pop("n_lags")
            final_params["lags"] = list(range(1, n_lags + 1))

        self.model_ = tsknn(**final_params)
        self.model_.fit(X)

        return self

    def predict(self, h):
        if not hasattr(self, "model_"):
            raise RuntimeError("You must fit the model before making predictions.")

        self.model_.h = h
        maxlags = np.max(self.model_.lags)
        X_history = self.model_.X[-maxlags:]
        return self.model_.predict(X_history)
