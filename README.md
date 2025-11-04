# tsknn

[![PyPI version](https://badge.fury.io/py/tsknn.svg)](https://badge.fury.io/py/tsknn)

*TSKNN* (Time Series K-Nearest Neighbors) is a Python implementation of the k-nearest neighbors (KNN) algorithm designed specifically for time series forecasting.

It is a pure Python and NumPy reimplementation of the [tsfknn](https://github.com/franciscomartinezdelrio/tsfknn) package from R—completely rewritten from scratch without relying on C extensions. By leveraging optimized NumPy operations, TSKNN achieves high performance while remaining lightweight and easy to install.


## Installation

```bash
pip install tsknn
```

Or clone the repository and install locally:

```bash
git clone https://github.com/ricardozago/tsknn.git
cd tsknn
pip install .
```

## Usage example

```python
import numpy as np

from tsknn import tsknn

X = np.random.rand(1000)

model = tsknn(
    k=[5, 7, 9], # number of neighbors to consider
    cf="mean",  # also median or weighted
    transform="multiplicative",  # also additive or None
    lags=[1, 2, 3, 4, 5, 6], # lags to consider as features
    distance="euclidean", # also manhattan
    h=24,  # forecast horizon
    msas="recursive",  # also mimo
)
model.fit(X)
tsknn_resp = model.predict()
print(tsknn_resp)
```

## AutoTSKNN: Automatic Hyperparameter Tuning

`autotsknn` automates the process of finding the best hyperparameters for your time series. It uses Bayesian optimization to search for the optimal combination of `k`, `cf`, `transform`, `distance`, and `lags`.

```python
import numpy as np
from tsknn.autotsknn import autotsknn

X = np.random.rand(1000)

model = autotsknn(
    metric="mse", # mse, mae, or mape
    n_trials=100, # number of optimization trials
    validation_splits=[0.1, 0.2, 0.3, 0.4] # list of validation splits to test
)
model.fit(X)
preds = model.predict(h=24)
print(preds)
```

## Tests

To run the tests and check coverage:

```bash
python -m pytest --cov=tsknn --cov-report term-missing
```

The full test suite provides 100% code coverage. To run all tests, you’ll need a working R installation, as the results are compared against the original `tsfknn` package. You’ll also need the `rpy2` library to interface between Python and R.

## Contributing

Contributions are welcome! Open issues or pull requests.

1. Fork the project
2. Create your branch (`git checkout -b feature/feature-name`)
3. Commit your changes (`git commit -am 'feat: new feature'`)
4. Push to the branch (`git push origin feature/feature-name`)
5. Open a Pull Request

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more details.
