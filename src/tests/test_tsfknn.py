import numpy as np
from itertools import product
from rpy2.robjects import r, pandas2ri, globalenv
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import IntVector, FloatVector
from numpy.testing import assert_almost_equal
from tsknn.tsknn import tsknn

np.set_printoptions(suppress=True)

np.random.seed(42)
X = np.random.rand(1000)


def tsfknn(X, h, transform, lags, msas, k):
    lags = IntVector(lags)
    r_X = FloatVector(X)

    r.assign("data", r_X);
    r.assign("h", h);
    r.assign("transform", transform);
    r.assign("lags", lags);
    r.assign("msas", msas);
    r.assign("k", k);

    r('''
    library(tsfknn)
    pred <- knn_forecasting(data,
                            h = h, 
                            k = k, 
                            msas = msas,
                            transform=transform,
                            lags = lags
    )

    pred_trat <- c(pred$prediction)
    ''')
    with localconverter(pandas2ri.converter):
        pred_py = globalenv['pred_trat']
    return pred_py

def test_tsfknn():
    # Define options
    msas_options = ["MIMO", "recursive"]
    transform_options = [None, "multiplicative", "additive"]
    lags_options = [[1, 2], [1, 2, 3], [1, 2, 3, 4], [3, 4, 5, 9, 10], [5, 15], [1, 4, 6, 12]]
    k_options = [2, 3, 5, 10, 25, 50, 250]
    h_options = [24, 48, 72, 100]

    # Iterate over all combinations of parameters
    for msas, transform, lags, k, h in product(msas_options, transform_options, lags_options, k_options, h_options):
        print(f"msas={msas}, transform={transform}, lags={lags}, k={k}, h={h}")

        # Train and predict using tsknn
        model = tsknn(cf="mean",
                      h=h,
                      transform=transform,
                      lags=lags,
                      msas=msas.lower(),
                      k=k,
                      force_stable=False
                      )
        model.fit(X)
        tsknn_resp = model.predict()

        # Adjust transform for tsfknn
        transform_ = transform if transform is not None else "none"

        # Predict using tsfknn
        tsfknn_resp = tsfknn(X, h=h, transform=transform_, lags=lags, msas=msas, k=k)

        # Validate results
        assert_almost_equal(tsknn_resp, tsfknn_resp, decimal=5)
