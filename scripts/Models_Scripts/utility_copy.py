import numpy as np
from datetime import datetime
import pandas as pd
from math import sqrt
import os
from sklearn.linear_model import LinearRegression
import statistics as st
from math import floor

class TimeSeries:
    """
    A class used to represent a TimeSeries for forecasting.
    ...
    Attributes
    ----------
    name : str
        A name for easy identification of the results (like the type of base stock)
    series : np.array
        The numerical data of the series
    date_index : pd.DatetimeIndex
        a pandas DatetimeIndex representing the index of the original pandas series
    train_window_size : int
        size of the window to be used for training the model
    test_window_size : int
        size of the window to be used for making predictions
    window : int
        total size of the window (train_window_size + test_window_size)
    n_predictions : int
        the number of predictions that can be made given the series and window sizes
    train_windows : np.array
        2D numpy array where each row is a training window from the series
    test_windows : np.array
        2D numpy array where each row is a prediction window from the series
    predictions : np.array
        2D numpy array to store the model's predictions
    residuals : np.array
        1D numpy array of residuals/errors where each value represents a window
    abs_residuals : np.array
        the absolute values of the residuals/errors
    abs_mean_error : float
        float representing the mean absolute error of the model's predictions over all the windows
    Methods
    -------
    create_windows():
        Creates training and prediction windows from the series using the values provided at object instantiation.
    mse():
        Evaluates the model's performance by calculating the mean absolute error.
    save_data(model, filepath='data.csv'):
        Saves the model's metadata and performance metrics to a CSV file.
    """

    def __init__(self, name, series, train_window_size, test_window_size, window_step, frequency, fillna, value_description):
        """
        Parameters
        ----------
            name : str
                A name for easy identification of the results (like the type of base stock)
            series : pd.Series
                The pandas series object with a datetime index
            train_window_size : int
                The size of the window for training the model
            test_window_size : int
                The size of the window for making predictions
            frequency: str
                Frequency of data observations: monthly, weekly, daily etc
            fillna: str
                Method used to fill NaN values
            value_description: str
                Midpoint, value high and value low
        """
        self.df = series
        self.frequency = frequency
        self.name = name
        self.series = series.values
        self.date_index = series.index
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.window = train_window_size + test_window_size
        self.fillna = fillna
        self.value_description = value_description

        ## calculating n_predictions 
        self.step = window_step
        total_step = len(self.series) - (self.train_window_size + self.test_window_size)
        self.total_step = total_step
        
        if int(total_step % window_step) == 0:
            n_predictions = int(total_step/window_step) + 1
        else: 
            n_predictions = floor(total_step/window_step) + 2
        
        
        

        self.n_predictions = n_predictions
    def create_windows(self):
        """
        Creates training and prediction windows from the series.
        Returns
        -------
        response : dict
            A dictionary containing training and prediction windows.
        """
        # Initialize the train and predict windows
        train_windows = np.zeros((self.n_predictions, self.train_window_size))
        test_windows = np.zeros((self.n_predictions, self.test_window_size))
        predictions = test_windows.copy() # Predictions are the same size as actuals

        step_array = []

        # Iterate over the series to populate the windows
        for i in range(self.n_predictions):
            if (i < self.n_predictions - 1) or (self.total_step % self.step == 0):
                train_windows[i] = self.series[(i*self.step):(i*self.step)+self.train_window_size]
                test_windows[i] = self.series[(i*self.step)+self.train_window_size: (i*self.step) + self.train_window_size+self.test_window_size]
            else: 
                train_windows[i] = self.series[-(self.train_window_size + self.test_window_size):-self.test_window_size]
                test_windows[i] = self.series[-self.test_window_size:]

        # Save windows as class attributes for easy access
        self.train_windows = train_windows
        self.test_windows = test_windows
        self.predictions = predictions


        # preparing a response. Because we're not animals
        response = dict(train_windows=train_windows, test_windows=test_windows)
        return response
    def create_index_windows(self):
        self.train_windows_index =  self.train_windows
        self.test_windows_index = self.test_windows


    def mean_absolute_error(self):
        """
        Evaluates the model's performance by calculating the mean absolute error.
        Returns
        -------
        str
            A string stating the mean absolute error of the model's predictions.
        """
        actuals = self.test_windows
        forecast = self.predictions

        # Calculates the error
        residuals = actuals - forecast


        # Need to do the sum inside the square root
        abs_residuals = np.abs(residuals)
        abs_mean_error = abs_residuals.mean()

        self.residuals = residuals
        self.abs_residuals = abs_residuals
        self.abs_mean_error = abs_mean_error

        return abs_mean_error
    def mean_squared_error(self):
        """
        MSE - Provides the mean absolute error for the model's predictions.
        """
        squared_residuals = self.residuals**2
        # não é a média, é n-1
        mse = squared_residuals.mean()
        self.mse = mse
        return mse
    
    def root_mean_squared_error(self):
        """
        Evaluates the model's performance by calculating the mean absolute error.
        Returns
        -------
        str
            A string stating the mean absolute error of the model's predictions.
        """

        rmse = np.sqrt(self.mean_squared_error())
        self.rmse = rmse
        return rmse


    def mean_absolute_percentage_error(self):
        """
        Evaluates the model's performance by calculating the mean absolute percentage error.
        Returns
        ---
        str
            A string stating the mean absolute percentage error of the model's predictions.
        """
        actuals = np.array(self.test_windows)
        forecast = self.predictions

        # Calculates the error
        residuals = actuals - forecast
       
        abs_residuals = np.abs(residuals)

        self.abs_residuals = abs_residuals

        percentage_error = np.abs(residuals / actuals)
        mape = percentage_error.mean()
        self.mape = mape
        return mape
    
    def error_1st_prediction(self):
        """
        MAE_1st, MSE_1st, RMSE_1st and MAPE_1st - Provides the mean absolute error and mean squared error
        for the model's 1st prediction.
        """
        residuals_1st = self.residuals[:, 0]
        actual_1st = self.test_windows[:, 0]
        abs_residuals_1st = np.abs(residuals_1st)
        mae_1st = abs_residuals_1st.mean()
        mse_1st = np.mean(residuals_1st**2)
        rmse_1st = sqrt(mse_1st)
        mape_1st = np.abs(residuals_1st / actual_1st)
        mape_1st = mape_1st.mean()
        return (rmse_1st, mape_1st, mae_1st, mse_1st)

    def error_2nd_prediction(self):
        """
        MAE_2nd, MSE_2nd - Provides the mean absolute error and mean squared error
        for the model's 2nd prediction.'
        """
        residuals_2nd = self.residuals[:, 1]
        actual_2nd = self.test_windows[:, 1]

        abs_residuals_2nd = np.abs(residuals_2nd)
        mae_2nd = abs_residuals_2nd.mean()
        mse_2nd = np.mean(residuals_2nd**2)
        rmse_2nd = sqrt(mse_2nd)
        mape_2nd = np.abs(residuals_2nd / actual_2nd )    
        mape_2nd = mape_2nd.mean()   
        return (rmse_2nd, mape_2nd, mae_2nd, mse_2nd)
    
    def error_3rd_prediction(self):
        """
        MAE_3rd, MSE_3rd - Provides the mean absolute error and mean squared error
        for the model's 3rd prediction.
        """
        residuals_3rd = self.residuals[:, 2]
        actual_3rd = self.test_windows[:, 2]

        abs_residuals_3rd = np.abs(residuals_3rd)
        mae_3rd = abs_residuals_3rd.mean()
        mse_3rd = np.mean(residuals_3rd**2)
        rmse_3rd = sqrt(mse_3rd)
        mape_3rd = np.abs(residuals_3rd / actual_3rd)
        mape_3rd = mape_3rd.mean()
        return (rmse_3rd,mape_3rd, mae_3rd, mse_3rd)
    
    def mean_directional_accuracy(self):
        """
        MDA - Provides the mean directional accuracy for the model's predictions.
        This measure shows the proportion of forecasts that correctly predict the direction of change.
        """
        actual_direction = np.sign(self.test_windows[1:] - self.test_windows[:-1])
        forecast_direction = np.sign(self.predictions[1:] - self.predictions[:-1])
        mda = np.mean(actual_direction == forecast_direction)
        return mda
    
    def mean_absolute_scaled_error(self):
        """
        MASE - Provides the mean absolute scaled error for the model's predictions.
        Assumes the naive forecasting method of the previous observation.
        """
        naive_forecast_residuals = self.series[self.train_window_size:-1] - self.series[self.train_window_size-1:-2]
        mae_naive = np.mean(np.abs(naive_forecast_residuals))
        mase = self.mae / mae_naive
        return mase
    

    def linear_regression(self):
        """
        returns the linear regression
        """
        reg = LinearRegression()
        Forescast_tendency = []
        Real_tendency = []
        # Series = self.series.copy()
        for i in range(len(self.predictions)):
            ## Forecast Tendency
            predict = self.predictions[i]
            train = self.train_windows[i]

            ForecastTendencyFittingArray = np.concatenate(([train[-1]],predict))
            ForecastTendencyFittingArray =  np.array(ForecastTendencyFittingArray).reshape(-1,1)
            index = np.array([x for x in range(len(ForecastTendencyFittingArray))]).reshape(-1,1)
            reg.fit(index,ForecastTendencyFittingArray)
            Forescast_tendency.append(reg.coef_)


            ## Real Tendency
            test = self.test_windows[i]
            TestTendencyFittingArray = np.concatenate(([train[-1]],test))
            TestTendencyFittingArray = np.array(TestTendencyFittingArray).reshape(-1,1)
            reg.fit(index,TestTendencyFittingArray)

            Real_tendency.append(reg.coef_)
        self.Real_tendency = Real_tendency
        self.Forescast_tendency = Forescast_tendency
        Real_tendency_sign = np.sign(Real_tendency)
        Forecast_tendency_sign = np.sign(Forescast_tendency)
        regression_mda = np.mean(Real_tendency_sign == Forecast_tendency_sign)
        return regression_mda
  


    def confidence_interval_1st(self):
        ## x = média +- t*(std_deviation)/ root(sample_size), t = 12.706 
        forecast = self.predictions[:,0][-12:]
        ## first_error
        mean = forecast.mean()
        std_deviation = st.stdev(forecast)
        sample_size = len(forecast)
        root_sample_size = sqrt(sample_size)
        upper_confidence_interval_95percent = mean + 2.201*std_deviation/root_sample_size
        lower_confidence_interval_95percent = mean - 2.201*std_deviation/root_sample_size
        upper_confidence_interval_99percent = mean + 3.106*std_deviation/root_sample_size
        lower_confidence_interval_99percent = mean - 3.106*std_deviation/root_sample_size

        return upper_confidence_interval_95percent, lower_confidence_interval_95percent, upper_confidence_interval_99percent, lower_confidence_interval_99percent
    
    def confidence_interval_2nd(self):
        ## x = média +- t*(std_deviation)/ root(sample_size), t = 12.706 
        forecast = self.predictions[:,1][-12:]
        ## first_error
        mean = forecast.mean()
        std_deviation = st.stdev(forecast)
        sample_size = len(forecast)
        root_sample_size = sqrt(sample_size)
        upper_confidence_interval_95percent = mean + 2.201*std_deviation/root_sample_size
        lower_confidence_interval_95percent = mean - 2.201*std_deviation/root_sample_size
        upper_confidence_interval_99percent = mean + 3.106*std_deviation/root_sample_size
        lower_confidence_interval_99percent = mean - 3.106*std_deviation/root_sample_size

        return upper_confidence_interval_95percent, lower_confidence_interval_95percent, upper_confidence_interval_99percent, lower_confidence_interval_99percent
    
    def confidence_interval_3rd(self):
        ## x = média +- t*(std_deviation)/ root(sample_size), t = 12.706 
        forecast = self.predictions[:,2][-12:]
        ## first_error
        mean = forecast.mean()
        std_deviation = st.stdev(forecast)
        sample_size = len(forecast)
        root_sample_size = sqrt(sample_size)
        upper_confidence_interval_95percent = mean + 2.201*std_deviation/root_sample_size
        lower_confidence_interval_95percent = mean - 2.201*std_deviation/root_sample_size
        upper_confidence_interval_99percent = mean + 3.106*std_deviation/root_sample_size
        lower_confidence_interval_99percent = mean - 3.106*std_deviation/root_sample_size

        return upper_confidence_interval_95percent, lower_confidence_interval_95percent, upper_confidence_interval_99percent, lower_confidence_interval_99percent
    


    def evaluate_model(self):
        """
        Evaluates the model's performance by calculating several metrics.
        """
        actuals = self.test_windows
        forecast = self.predictions
        self.residuals = actuals - forecast
        
        self.mae = round(self.mean_absolute_error(),3)
        self.mse = round(self.mean_squared_error(),3)
        self.rmse = round(self.root_mean_squared_error(),3)
        self.rmse_1st = round(self.error_1st_prediction()[0],3)
        self.mape_1st = round(self.error_1st_prediction()[1],3)
        self.mae_1st = round(self.error_1st_prediction()[2],3)
        self.mse_1st = round(self.error_1st_prediction()[3],3)
        self.rmse_2nd = round(self.error_2nd_prediction()[0],3)
        self.mape_2nd = round(self.error_2nd_prediction()[1],3)
        self.mae_2nd = round(self.error_2nd_prediction()[2],3)
        self.mse_2nd = round(self.error_2nd_prediction()[3],3)
        self.rmse_3rd = round(self.error_3rd_prediction()[0],3)
        self.mape_3rd = round(self.error_3rd_prediction()[1],3)
        self.mae_3rd = round(self.error_3rd_prediction()[2],3)
        self.mse_3rd = round(self.error_3rd_prediction()[3],3)
        self.mape = round(self.mean_absolute_percentage_error(),3)
        self.mase = round(self.mean_absolute_scaled_error(),3)
        self.mda = round(self.mean_directional_accuracy(),3)
        self.regression_mda  = self.linear_regression()
        self.upper_confidence_interval_95percent_1st = round(self.confidence_interval_1st()[0],3)
        self.lower_confidence_interval_95percent_1st = round(self.confidence_interval_1st()[1],3)
        self.upper_confidence_interval_99percent_1st = round(self.confidence_interval_1st()[2],3)
        self.lower_confidence_interval_99percent_1st = round(self.confidence_interval_1st()[3],3)
        self.upper_confidence_interval_95percent_2nd = round(self.confidence_interval_2nd()[0],3)
        self.lower_confidence_interval_95percent_2nd = round(self.confidence_interval_2nd()[1],3)
        self.upper_confidence_interval_99percent_2nd = round(self.confidence_interval_2nd()[2],3)
        self.lower_confidence_interval_99percent_2nd = round(self.confidence_interval_2nd()[3],3)
        self.upper_confidence_interval_95percent_3rd = round(self.confidence_interval_3rd()[0],3)
        self.lower_confidence_interval_95percent_3rd = round(self.confidence_interval_3rd()[1],3)
        self.upper_confidence_interval_99percent_3rd = round(self.confidence_interval_3rd()[2],3)
        self.lower_confidence_interval_99percent_3rd = round(self.confidence_interval_3rd()[3],3)
    
    
        return {
            "MAE": self.mae, 
            "MSE": self.mse, 
            "RMSE": self.rmse,
            "MAE_1st": self.mae_1st,
            "MSE_1st": self.mse_1st,  
            "RMSE_1st": self.rmse_1st, 
            "MAPE_1st": self.mape_1st,  
            "MAE_2nd": self.mae_2nd,
            "MSE_2nd": self.mse_2nd, 
            "RMSE_2nd": self.rmse_2nd,
            "MAPE_2nd": self.mape_2nd,
            "MAE_3rd": self.mae_3rd,
            "MSE_3rd": self.mse_3rd,
            "RMSE_3rd": self.rmse_3rd, 
            "MAPE_3rd": self.mape_3rd,
            "MAPE": self.mape, 
            "MASE": self.mase, 
            "MDA": self.mda, 
            'upper_confidence_interval_95percent_1st': self.upper_confidence_interval_95percent_1st,
            'lower_confidence_interval_95percent_1st': self.lower_confidence_interval_95percent_1st,
            'upper_confidence_interval_99percent_1st': self.upper_confidence_interval_99percent_1st,
            'lower_confidence_interval_99percent_1st': self.lower_confidence_interval_99percent_1st,
            'upper_confidence_interval_95percent_2nd': self.upper_confidence_interval_95percent_2nd,
            'lower_confidence_interval_95percent_2nd': self.lower_confidence_interval_95percent_2nd,
            'upper_confidence_interval_99percent_2nd': self.upper_confidence_interval_99percent_2nd,
            'lower_confidence_interval_99percent_2nd': self.lower_confidence_interval_99percent_2nd,
            'upper_confidence_interval_95percent_3rd': self.upper_confidence_interval_95percent_3rd,
            'lower_confidence_interval_95percent_3rd': self.lower_confidence_interval_95percent_3rd,
            'upper_confidence_interval_99percent_3rd': self.upper_confidence_interval_99percent_3rd,
            'lower_confidence_interval_99percent_3rd': self.lower_confidence_interval_99percent_3rd
        }  


    def save_data(self, model,PRICE_TYPE_DESCRIPTION, filepath='models_results.csv'):
        """
        Saves the model's metadata and performance metrics to a CSV file.
        Parameters
        ----------
        model : str
            The name or type of the model used.
        filepath : str, optional
            The file path where the data will be saved (default is 'data.csv')
        Returns
        -------
        df : pd.DataFrame
            A pandas DataFrame containing the saved data.
        """

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        data = dict(
            name=[self.name],
            model=[model],
            n_predictions=[self.n_predictions],
            window_step = [self.step],
            frequency=[self.frequency],
            PRICE_TYPE_DESCRIPTION=[PRICE_TYPE_DESCRIPTION],
            train_window_size=[self.train_window_size],
            test_window_size=[self.test_window_size],
            mean_absolute_error=[self.mae],
            mape=[self.mape],
            rmse=[self.rmse],
            rmse_1st = [self.rmse_1st],
            rmse_2nd = [self.rmse_2nd],
            rmse_3rd = [self.rmse_3rd],
            mape_1st = [self.mape_1st],
            mape_2nd = [self.mape_2nd],
            mape_3rd = [self.mape_3rd],
            mda = [self.mda],
            regression_mda = [self.regression_mda],
            upper_confidence_interval_95percent_1st= [self.upper_confidence_interval_95percent_1st],
            lower_confidence_interval_95percent_1st= [self.lower_confidence_interval_95percent_1st],
            upper_confidence_interval_99percent_1st= [self.upper_confidence_interval_99percent_1st],
            lower_confidence_interval_99percent_1st= [self.lower_confidence_interval_99percent_1st],
            upper_confidence_interval_95percent_2nd= [self.upper_confidence_interval_95percent_2nd],
            lower_confidence_interval_95percent_2nd= [self.lower_confidence_interval_95percent_2nd],
            upper_confidence_interval_99percent_2nd= [self.upper_confidence_interval_99percent_2nd],
            lower_confidence_interval_99percent_2nd= [self.lower_confidence_interval_99percent_2nd],
            upper_confidence_interval_95percent_3rd= [self.upper_confidence_interval_95percent_3rd],
            lower_confidence_interval_95percent_3rd= [self.lower_confidence_interval_95percent_3rd],
            upper_confidence_interval_99percent_3rd= [self.upper_confidence_interval_99percent_3rd],
            lower_confidence_interval_99percent_3rd= [self.lower_confidence_interval_99percent_3rd],
            fillna = [self.fillna],
            timestamp=[now]
        )

        df = pd.DataFrame(data=data)
        file_exists = os.path.exists(filepath)
        # Appends data to output file if it exists. Otherwise, creates it.
        if file_exists:
            df.to_csv(filepath, index=False, mode='a', header=False)
        else:
            df.to_csv(filepath, index=False)

        return df












        
