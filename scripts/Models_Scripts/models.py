import pandas as pd
import numpy as np
import seaborn as sns
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
from statsmodels.tsa.api import ExponentialSmoothing
import statsmodels.api as sm

class UFPR_Models:
    """
    A scrap of a class for evaluating the models. 
    ...
    Atributes: 
    ----------
    
    series: pandas.DataFrame
        A dataframe indexed by date and a column with the data we want to predict

    train_base: list
        The serie values that the model will use to train

    test_base: list
        The serie values that we want the model to forecast (values out of the training base)

    params_grid: dict
        Set the p, d and q ranges to search for the best params to fit arima model given the serie to forecast
    
    frequency: str
        Groups the serie by month or week ('m' for month and 'w' for week)
    """
    def __init__(self,  evaluate_params: tuple=None, train_base:pd.Series = None, train_base_index: list = None) -> None:
        self.train_base = train_base
        self.evaluate_params = evaluate_params
        self.train_base_index = train_base_index
        

    def Evaluate_Arima(self) -> dict:
        warnings.filterwarnings("ignore")


        
        self.predictions = list()
        self.fitting = list()

            
        model = ARIMA(self.train_base, order=self.evaluate_params)
        model_fit = model.fit()

        self.predictions.append(model_fit.forecast(3)[:3])
        self.fitting.append(model_fit.predict())


    def Simple_Exponential_Smoothing(self):
        # Ajustar modelo de suavização exponencial simples
        ses_model = ExponentialSmoothing(self.train_base).fit()

        # Fazer previsões
        self.ses_forecast = ses_model.forecast(steps=3)

    def Linear_Regression(self):
        time = np.arange(len(self.train_base))
        time = sm.add_constant(time)
        regression_model = sm.OLS(self.train_base, time)
        fit = regression_model.fit()
        
        ## fazendo a previsão dos próximos 3 valores
        next_values = fit.predict(sm.add_constant(np.arange(len(self.train_base), len(self.train_base) + 3)))
        self.Linear_Regression_Forecast = next_values
        
    def Holt_Exponential_Smoothing(self, trend="add"):
        holt_model = ExponentialSmoothing(self.train_base, trend=trend).fit()
        holt_forecast = holt_model.forecast(steps=3)
        self.holt_forecast = holt_forecast

    def Holt_Winters(self, seasonal_decompose='add', freq=12):
        Series = pd.Series(data = self.train_base, index=self.train_base_index)
        # Series.index = pd.date_range(start=f'{self.train_base_index[0]}', end=f'{self.train_base_index[-1]}', freq="M")
        hw_model = ExponentialSmoothing(Series, seasonal=seasonal_decompose, seasonal_periods=freq).fit()
        hw_forecast = hw_model.forecast(steps=3)
        self.holt_winters_forecast = hw_forecast

