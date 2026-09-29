class InferencePipeline:
    def __init__(self, model=None):
        self.model = model

    def run(self, x):
        if self.model is None:
            raise ValueError("Model not loaded.")
        return self.model(x)
