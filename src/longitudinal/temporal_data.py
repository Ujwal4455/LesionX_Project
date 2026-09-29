class TemporalDataset:
    def __init__(self, data=None):
        self.data = data or []

    def __len__(self):
        return len(self.data)
