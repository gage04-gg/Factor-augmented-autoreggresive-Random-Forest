import numpy as np
def create_X(data, target, lags=2):
    X = []
    y = data[target].values
    for t in range(lags, len(data)):
        row = [1]
        for l in range(1, lags+1):
            row.append(y[t-l])
        X.append(row)
    return np.array(X)
def expanding_window_forecast(X, y, S, model, start_train=100):
    preds = []
    actuals = []
    for t in range(start_train, len(y)-1):
        X_train = X[:t]
        y_train = y[:t]
        S_train = S[:t]
        X_test = X[t:t+1]
        S_test = S[t:t+1]
        y_test = y[t]
        model.fit(X_train, y_train, S_train)
        y_pred = model.predict(X_test, S_test)[0]
        preds.append(y_pred)
        actuals.append(y_test)
        print(f"Step {t} done")
    return np.array(preds), np.array(actuals)
def extract_betas_over_time(X, y, S, model, start_train=100):
    betas_all = []
    for t in range(start_train, len(X)):
        model.fit(X[:t], y[:t], S[:t])
        beta_t = model.get_betas(X[t:t+1], S[t:t+1])
        betas_all.append(beta_t[0])
    return np.array(betas_all)