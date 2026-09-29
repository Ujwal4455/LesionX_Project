class SHAPExplainer:
    def __init__(self, model=None):
        self.model = model

    def explain(self, x):
        if self.model is None:
            return {"status": "unavailable", "message": "No model provided"}
        return {"status": "not_run", "message": "SHAP explanation requires a fitted model and dataset context"}
