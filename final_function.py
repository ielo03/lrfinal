import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score


def get_data():
    return pd.read_feather("initial_data.feather")


def regression(X, y, mode='predictive', alpha_values=None, l1_ratios=None, top_n=100, decay_rate=0.1, test_size=0.2,
               random_state=42):
    # Default alpha and l1_ratio values for Ridge, Lasso, and ElasticNet
    if alpha_values is None:
        alpha_values = [0.1, 1, 10, 100]
    if l1_ratios is None:
        l1_ratios = [0.2, 0.5, 0.8]  # ElasticNet l1_ratio defaults

    X = X.dropna()
    y = y.dropna()
    common_indices = X.index.intersection(y.index)
    X = X.loc[common_indices]
    y = y.loc[common_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if mode == 'predictive':
        # Train models
        ridge = RidgeCV(alphas=alpha_values, cv=5).fit(X_train, y_train)
        lasso = LassoCV(alphas=alpha_values, cv=5).fit(X_train, y_train)
        elastic_net = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratios, cv=5).fit(X_train, y_train)

        # Compare CV scores to select the best model
        ridge_score = np.mean(cross_val_score(ridge, X_train, y_train, cv=5))
        lasso_score = np.mean(cross_val_score(lasso, X_train, y_train, cv=5))
        elastic_net_score = np.mean(cross_val_score(elastic_net, X_train, y_train, cv=5))

        scores = {'Ridge': ridge_score, 'Lasso': lasso_score, 'ElasticNet': elastic_net_score}
        best_model_name = max(scores, key=scores.get)
        best_model = {'Ridge': ridge, 'Lasso': lasso, 'ElasticNet': elastic_net}[best_model_name]

    elif mode == 'explanatory':
        # Use Ordinary Least Squares for interpretability
        best_model = LinearRegression().fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    diagnostics = {}
    if mode == 'explanatory':
        diagnostics['coefficients'] = pd.Series(best_model.coef_, index=X.columns)
        diagnostics['residuals'] = y_test - y_pred

    return {
        'model': best_model,
        'best_model_name': best_model_name if mode == 'predictive' else 'OLS',
        'mse': mse,
        'r2': r2,
        'diagnostics': diagnostics if mode == 'explanatory' else None,
        'predictions': y_pred,
        'actual': y_test,
    }