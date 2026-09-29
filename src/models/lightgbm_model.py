import lightgbm as lgb


class LightGBMClassifier:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        self.model = None

    def fit(self, X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(self.params, train_data, num_boost_round=200)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)
