import numpy as np
from tsengine.data import generate_ar_series
from tsengine.diagnostics import adf_test
from tsengine.ar_model import ARModel
from tsengine.evaluation import rmse

def main():
    series = generate_ar_series(phi=[0.6, -0.2], n=600, sigma=1.0, seed=1)
    print("Diagnostics:", adf_test(series))

    train = series[:500]
    test = series[500:520]

    model = ARModel(p=2).fit_ols(train)
    pred = model.forecast(train, steps=len(test))

    print("RMSE:", round(rmse(test, pred), 4))

if __name__ == "__main__":
    main()

