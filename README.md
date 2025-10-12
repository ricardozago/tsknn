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
    cf="mean",  # also median or weighted
    h=24,  # forecast horizon
    transform="multiplicative",  # also additive or None
    msas="recursive",  # also mimo
    lags=[1, 2, 3, 4, 5, 6], # lags to consider as features
    k=[5, 7, 9], # number of neighbors to consider
)
model.fit(X)
tsknn_resp = model.predict()
print(tsknn_resp)
```

## Tests

To run the tests and check coverage:

```bash
python -m pytest --cov=tsknn --cov-report term-missing
```

To run all tests, you’ll need R installed, since the results are compared against the original tsfknn package. You’ll also need the rpy2 library to interface between Python and R.

## Contributing

Contributions are welcome! Open issues or pull requests.

1. Fork the project
2. Create your branch (`git checkout -b feature/feature-name`)
3. Commit your changes (`git commit -am 'feat: new feature'`)
4. Push to the branch (`git push origin feature/feature-name`)
5. Open a Pull Request

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more details.
