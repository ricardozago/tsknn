# tsknn

`tsknn` is a Python implementation of the k-nearest neighbors (KNN) algorithm aimed at time series forecasting. The package provides utilities for lag selection, hyperparameter optimisation and multiple forecasting strategies.

## Key features

- Support for several distance metrics: `euclidean`, `manhattan`, `chebyshev`, `cosine` and `dtw`.
- Different aggregation schemes for neighbours (`cf`): mean, median, weighted or trimmed.
- MIMO, recursive and direct forecasting modes (`msas`).
- Additive or multiplicative transformations for trend removal.
- `optimize_params` helper to search for the best hyperparameters.
- Cross validation via `cross_validate_params` to choose `k` and `lags`.
- Optimisation by *grid search*, *random search* or Bayesian approach.
- Optional handling of missing values with the `nan_strategy` argument.
- Weight neighbours by recency or real distance through `weight_by`.
- Multivariate forecasting via the `tsknn` class.

## Installation

```bash
pip install tsknn
```

For local development clone the repository and install it in editable mode:

```bash
pip install -e .
```

## Quick example

The example below uses the air passengers data in `data/AirPassengers.csv` to forecast the next 12 months.

```python
import pandas as pd
from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values

lags = 3
X_pred = X[-lags:]
model = tsknn(k=3, h=12, transform="multiplicative", lags=lags,
              nan_strategy="interpolate")
model.fit(X)
forecast = model.predict(X_pred)
print(forecast)
```
A full script can be found at `examples/knn_example.py`.

`tsknn` also works with multivariate inputs:

```python
from tsknn import tsknn
df["passengers2"] = df["passengers"] * 1.1
X_multi = df[["passengers", "passengers2"]].values
model = tsknn(k=3, h=12, lags=3)
model.fit(X_multi)
forecast = model.predict(X_multi[-3:])
print(forecast)
```

## Hyperparameter optimisation

Use `optimize_params` to find the best combination of parameters. Provide a `param_grid` with the values to test and a validation series. The metric can be passed as name or function.
You can also define the search method with the `method` argument.

```python
from tsknn import optimize_params

param_grid = {
    "k": [2, 3, 4],
    "lags": [3, 5],
    "h": [12]
}
# choose the metric among "rmse", "mae" or "mape"
best, score = optimize_params(X, param_grid, metric="mae", method="random", n_iter=10)
print(best, score)

To evaluate combinations across multiple splits of the series use `cross_validate_params`:

```python
from tsknn import cross_validate_params

best, score = cross_validate_params(X, param_grid, n_splits=3)
print(best, score)
```

## Automatic model selection

To automate the search and return a trained model, use `autotsknn`.
It evaluates different combinations of `k` and `lags` and returns the best model.
```python
from tsknn import autotsknn

# runs the search using default values for k and lags
model, params, score = autotsknn(X, h=12, search_method="bayes", n_iter=15)
print(params, score)
```

## Additional examples

The `examples/` folder contains complete scripts demonstrating different usage flows:
- `knn_example.py` – direct execution of `tsknn` forecasting 5 values.
- `optimize_params_example.py` – usage of `optimize_params` to search for the best hyperparameters.
- `autotsknn_example.py` – automatic selection of `k` and `lags` with Bayesian optimisation.

Run the scripts with Python to see the results in action, for example:

```bash
python examples/optimize_params_example.py
```

## Model persistence

`tsknn` models can be saved to disk and loaded back:

```python
model = tsknn(lags=3, h=2)
model.fit(X)
model.save("model.pkl")

loaded = tsknn.load("model.pkl")
preds = loaded.predict(X[-3:])
```

## scikit-learn compatibility

`tsknn` implements the `BaseEstimator` and `RegressorMixin` APIs so it can be
used in scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from tsknn import tsknn

pipeline = Pipeline([("model", tsknn(lags=3, h=2))])
pipeline.fit(X)
preds = pipeline.predict(X[-3:])
```

## Tests
Unit tests can be run with `pytest` after installing the development dependencies:

```bash
pip install -r requirements-dev.txt
pytest
```

## Licence

Distributed under the MIT licence. See the `LICENSE` file for further information.
