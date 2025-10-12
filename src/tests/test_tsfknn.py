import numpy as np
from itertools import product
from rpy2.robjects import r, pandas2ri, globalenv
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import IntVector, FloatVector
from numpy.testing import assert_almost_equal
from tsknn import tsknn

np.set_printoptions(suppress=True)

np.random.seed(42)
X = np.random.rand(1000)


def tsfknn(X, h, transform, lags, msas, k, cf):
    lags = IntVector(lags)
    r_X = FloatVector(X)

    r.assign("data", r_X);
    r.assign("h", h);
    r.assign("transform", transform);
    r.assign("lags", lags);
    r.assign("msas", msas);
    r.assign("k", k);
    r.assign("cf", cf);
    

    r('''
    library(tsfknn)
    pred <- knn_forecasting(data,
                            h = h, 
                            k = k, 
                            msas = msas,
                            transform=transform,
                            lags = lags,
                            cf = cf
    )

    pred_trat <- c(pred$prediction)
    ''')
    with localconverter(pandas2ri.converter):
        pred_py = globalenv['pred_trat']
    return pred_py


def test_tsfknn():
    cf_options = ["mean", "median"] #weighted
    msas_options = ["MIMO", "recursive"]
    transform_options = [None, "additive", "multiplicative"]
    lags_options = [[1, 2], [1, 2, 3], [1, 2, 3, 4], [3, 4, 5, 9, 10], [1, 4, 6, 12]]
    k_options = [2, 3, 5, 10, 25]
    h_options = [24, 36]

    # Iterate over all combinations of parameters
    for msas, transform, lags, k, h, cf in product(msas_options, transform_options, lags_options, k_options, h_options, cf_options):
        # Train and predict using tsknn
        model = tsknn(cf=cf, h=h, transform=transform, lags=lags, msas=msas.lower(), k=k, force_stable=True)
        model.fit(X)
        tsknn_resp = model.predict()

        # Adjust transform for tsfknn
        transform_ = transform if transform is not None else "none"

        # Predict using tsfknn
        tsfknn_resp = tsfknn(X, h=h, transform=transform_, lags=lags, msas=msas, k=k, cf=cf)

        # Validate results
        assert_almost_equal(tsknn_resp, tsfknn_resp, decimal=5)


def test_tsfknn_weighted():
    # Testar weighted é complicado, pois quando uma das distâncias é muito próxima de zero, o resultado pode divergir da tsfknn
    cf_options = ["weighted"]
    msas_options = ["MIMO", "recursive"]
    transform_options = [None, "additive", "multiplicative"]
    lags_options = [[1, 2], [1, 2, 3]]
    k_options = [2, 3]
    h_options = [6]

    # Iterate over all combinations of parameters
    for msas, transform, lags, k, h, cf in product(msas_options, transform_options, lags_options, k_options, h_options, cf_options):
        print(f"msas={msas}, transform={transform}, lags={lags}, k={k}, h={h}, cb={cf}")

        # Train and predict using tsknn
        model = tsknn(cf=cf, h=h, transform=transform, lags=lags, msas=msas.lower(), k=k, force_stable=True)
        model.fit(X)
        tsknn_resp = model.predict()

        # Adjust transform for tsfknn
        transform_ = transform if transform is not None else "none"

        # Predict using tsfknn
        tsfknn_resp = tsfknn(X, h=h, transform=transform_, lags=lags, msas=msas, k=k, cf=cf)

        # Validate results
        assert_almost_equal(tsknn_resp, tsfknn_resp, decimal=5)
