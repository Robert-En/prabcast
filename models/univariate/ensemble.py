import pandas as pd


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, data):
        for model in self.models:
            model.fit(data)

    def predict(self, steps):
        forecasts = pd.DataFrame()
        for model in self.models:
            forecasts[model.__class__.__name__] = model.predict(steps)
        ensemble_forecast = forecasts.mean(axis=1)
        return ensemble_forecast
