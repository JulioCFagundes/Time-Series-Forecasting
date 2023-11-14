
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Classical_Statistics_Models.classical_statistics_models import UFPR_Models
from Machine_Learning_Models.RBF_model import rbf_model
from Machine_Learning_Models.kmeans_RBF import kmeans_rbf
from Classical_Statistics_Models.Baseline import random_walk


def mean_absolute_error(real_values, forecast):
    """
    Evaluates the model's performance by calculating the mean absolute error.
    Returns
    -------
    str
        A string stating the mean absolute error of the model's predictions.
    """

    # Calculates the error
    residuals = real_values - forecast
    # Need to do the sum inside the square root
    abs_residuals = np.abs(residuals)
    abs_mean_error = abs_residuals.mean()
  
    return abs_mean_error

def mean_squared_error(real_values, forecast):
    """
    MSE - Provides the mean absolute error for the model's predictions.
    """
    residuals = real_values - forecast
    squared_residuals = residuals**2
    # não é a média, é n-1
    mse = squared_residuals.mean()
    
    return mse

def root_mean_squared_error(real_values, forecast):
    """
    Evaluates the model's performance by calculating the mean absolute error.
    Returns
    -------
    str
        A string stating the mean absolute error of the model's predictions.
    """
    rmse = np.sqrt(mean_squared_error(real_values, forecast))
    return rmse

def mean_absolute_percentage_error(real_values, forecast):
    """
    Evaluates the model's performance by calculating the mean absolute percentage error.
    Returns
    ---
    str
        A string stating the mean absolute percentage error of the model's predictions.
    """

    # Calculates the error
    residuals = real_values - forecast
   
    percentage_error = np.abs(residuals / real_values)
    mape = percentage_error.mean()
    return mape
    

#%%

dataset_path =  r'C:\Users\User\Desktop\Exxon\Forecasting ARGUS base\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\scripts\Data_Pre_Processing\Cleaned_Datasets\argus_clean.csv'
dataset = pd.read_csv(dataset_path)



train_window_size = 48
test_window_size = 3
## name of the base oils

value_low_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value low'].copy()
value_high_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value high'].copy()
midpoint_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'midpoint'].copy()

products = list(dataset['CODE_DISPLAY_NAME'].unique())
value_high_products = list(value_high_df['CODE_DISPLAY_NAME'].unique())
value_low_products = list(value_low_df['CODE_DISPLAY_NAME'].unique())
midpoint_products = list(midpoint_df['CODE_DISPLAY_NAME'].unique())
print(len(value_high_products))

for product in products:
    value_low_df[value_low_df.CODE_DISPLAY_NAME == product] = value_low_df[value_low_df.CODE_DISPLAY_NAME == product]
    value_high_df[value_high_df.CODE_DISPLAY_NAME == product] = value_high_df[value_high_df.CODE_DISPLAY_NAME == product]
    midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product] = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product]
    if (product in value_low_df['CODE_DISPLAY_NAME'].values) and (len(value_low_df[value_low_df.CODE_DISPLAY_NAME == product].VALUE.values) <= train_window_size + test_window_size):
        value_low_products.remove(product)
    if (product in value_high_df['CODE_DISPLAY_NAME'].values) and (len(value_high_df[value_high_df.CODE_DISPLAY_NAME == product].VALUE.values) <= (train_window_size + test_window_size)):
        value_high_products.remove(product)
    if (product in midpoint_df['CODE_DISPLAY_NAME'].values) and (len(midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product].VALUE.values) <= train_window_size + test_window_size):
        midpoint_products.remove(product)


#%%

########################################################################
###########################  Vertical DF  ############################## 
iterations = 1

results_dictionary = dict()
arima_dictionary = dict()
for product_name in value_high_products:
    dataset_name = product_name
    serie = value_high_df[value_high_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    product_array = [product_name for i in range(3)]
    value_type = ['value_high' for i in range(3)]
    real_values = test_base

    baseline = random_walk(data=train_base.values, step=1)
    baseline_forecast = baseline.predict()
    baseline_error = real_values - baseline_forecast
    baseline_rmse = [root_mean_squared_error(real_values, baseline_forecast) for i in range(3)]
    baseline_mape = [mean_absolute_percentage_error(real_values, baseline_forecast) for i in range(3)]

    models = UFPR_Models(train_base=train_base.values.reshape(len(train_base.values)))
    models.Simple_Exponential_Smoothing()
    simple_exponential_smoothing_forecast = models.ses_forecast
    simple_exponential_smoothing_error = real_values - simple_exponential_smoothing_forecast
    simple_exponential_smoothing_rmse = [root_mean_squared_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]
    simple_exponential_smoothing_mape = [mean_absolute_percentage_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]

    models.Linear_Regression()
    Linear_Regression_Forecast = models.Linear_Regression_Forecast
    Linear_Regression_Error = real_values - Linear_Regression_Forecast
    Linear_Regression_rmse = [root_mean_squared_error(real_values, Linear_Regression_Forecast) for i in range(3)]
    Linear_Regression_mape = [mean_absolute_percentage_error(real_values, Linear_Regression_Forecast) for i in range(3)]

    trend = 'add'
    models.Holt_Exponential_Smoothing(trend)
    holt_add_forecast =  models.holt_forecast
    holt_add_error = real_values - holt_add_forecast
    holt_add_rmse = [root_mean_squared_error(real_values, holt_add_forecast) for i in range(3)]
    holt_add_mape = [mean_absolute_percentage_error(real_values, holt_add_forecast) for i in range(3)]

    trend = 'mul'
    models.Holt_Exponential_Smoothing(trend)
    holt_mul_forecast =  models.holt_forecast
    holt_mul_error =  real_values - holt_mul_forecast
    holt_mul_rmse = [root_mean_squared_error(real_values, holt_mul_forecast) for i in range(3)]
    holt_mul_mape = [mean_absolute_percentage_error(real_values, holt_mul_forecast) for i in range(3)]

    decomposition_type = 'add'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_add_forecast = models.holt_winters_forecast
    holt_winters_add_error = real_values - holt_winters_add_forecast
    holt_winters_add_rmse =[root_mean_squared_error(real_values, holt_winters_add_forecast) for i in range(3)]
    holt_winters_add_mape = [mean_absolute_percentage_error(real_values, holt_winters_add_forecast) for i in range(3)]

    decomposition_type = 'mul'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_mult_forecast = models.holt_winters_forecast
    holt_winters_mult_error = real_values - holt_winters_mult_forecast
    holt_winters_mult_rmse = [root_mean_squared_error(real_values, holt_winters_mult_forecast) for i in range(3)]
    holt_winters_mult_mape = [mean_absolute_percentage_error(real_values, holt_winters_mult_forecast) for i in range(3)]


    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param
    arima = UFPR_Models(train_base=train_base.values)

    for p in p_value:
        for d in d_value:
            for q in q_value:
                params = (p,d,q)
                
                try:
                    arima.Evaluate_Arima(params)
                    arima_predictions = arima.predictions[0].reshape(len(arima.predictions[0]))
                    if iterations == 1:
                        arima_dictionary[f'arima({p},{d},{q}) forecast'] = arima_predictions
                        arima_dictionary[f'arima({p},{d},{q}) error'] = real_values - arima_predictions
                        arima_dictionary[f'arima({p},{d},{q}) rmse'] = [root_mean_squared_error(real_values, arima_predictions) for i in range(3)]
                        arima_dictionary[f'arima({p},{d},{q}) mape'] = [mean_absolute_percentage_error(real_values, arima_predictions) for i in range(3)]
                    else:
                        arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],arima_predictions)
                        arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'], (real_values - arima_predictions))
                        arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], [root_mean_squared_error(real_values, arima_predictions) for i in range(3)])
                        arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],[mean_absolute_percentage_error(real_values, arima_predictions) for i in range(3)])
                except:
                    print('LU_decomposition error')
                    if iterations == 1:
                        arima_dictionary[f'arima({p},{d},{q}) forecast'] = ['decomposition error' for i in range(3)]
                        arima_dictionary[f'arima({p},{d},{q}) error'] = ['decomposition error' for i in range(3)]
                        arima_dictionary[f'arima({p},{d},{q}) rmse'] =  ['decomposition error' for i in range(3)]
                        arima_dictionary[f'arima({p},{d},{q}) mape'] = ['decomposition error' for i in range(3)]
            

                    else:
                        arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],['decomposition error' for i in range(3)])
                        arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'],['decomposition error' for i in range(3)])
                        arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], ['decomposition error' for i in range(3)])
                        arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],['decomposition error' for i in range(3)])
    
    if iterations == 1:

        results_dictionary[f'CODE_DISPLAY_NAME'] = product_array
    
        results_dictionary[f'value_type_description'] = value_type
        Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
        
        results_dictionary[f'Month_end'] = Month_end        
        results_dictionary[f'real_values'] = real_values


        results_dictionary[f'baseline_forecast'] = baseline_forecast
        results_dictionary[f'baseline_error'] = baseline_error       
        results_dictionary[f'baseline_rmse'] = baseline_rmse
        results_dictionary[f'baseline_mape'] = baseline_forecast
        
    
        results_dictionary[f'simple_exponential_smoothing_forecast'] = simple_exponential_smoothing_forecast
        results_dictionary[f'simple_exponential_smoothing_error'] = simple_exponential_smoothing_error
        results_dictionary[f'simple_exponential_smoothing_rmse'] = simple_exponential_smoothing_rmse
        results_dictionary[f'simple_exponential_smoothing_mape'] = simple_exponential_smoothing_error

        results_dictionary[f'Linear_Regression_Forecast'] = Linear_Regression_Forecast
        results_dictionary[f'Linear_Regression_Error'] = Linear_Regression_Error
        results_dictionary[f'Linear_Regression_rmse'] = Linear_Regression_rmse
        results_dictionary[f'Linear_Regression_mape'] = Linear_Regression_mape

        results_dictionary[f'holt_add_forecast'] = holt_add_forecast
        results_dictionary[f'holt_add_error'] = holt_add_error
        results_dictionary[f'holt_add_rmse'] = holt_add_rmse
        results_dictionary[f'holt_add_mape'] = holt_add_mape

        results_dictionary[f'holt_mul_forecast'] = holt_mul_forecast
        results_dictionary[f'holt_mul_error'] = holt_mul_error
        results_dictionary[f'holt_mul_rmse'] = holt_mul_rmse
        results_dictionary[f'holt_mul_mape'] = holt_mul_mape

        results_dictionary[f'holt_winters_add_forecast'] = holt_winters_add_forecast
        results_dictionary[f'holt_winters_add_error'] = holt_winters_add_error
        results_dictionary[f'holt_winters_add_rmse'] = holt_winters_add_rmse
        results_dictionary[f'holt_winters_add_mape'] = holt_winters_add_error

        results_dictionary[f'holt_winters_mult_forecast'] = holt_winters_mult_forecast
        results_dictionary[f'holt_winters_mult_error'] = holt_winters_mult_error
        results_dictionary[f'holt_winters_mult_rmse'] = holt_winters_mult_rmse
        results_dictionary[f'holt_winters_mult_mape'] = holt_winters_mult_mape

    else: 

        results_dictionary[f'CODE_DISPLAY_NAME'] =np.append(results_dictionary[f'CODE_DISPLAY_NAME'], product_array)
    
        results_dictionary[f'real_values'] = np.append(results_dictionary[f'real_values'], real_values)
        results_dictionary[f'value_type_description'] = np.append(results_dictionary[f'value_type_description'], value_type)
        results_dictionary[f'Month_end'] = np.append(results_dictionary[f'Month_end'], Month_end)


        results_dictionary[f'baseline_forecast'] = np.append(results_dictionary[f'baseline_forecast'],baseline_forecast)
        results_dictionary[f'baseline_error'] = np.append(results_dictionary[f'baseline_error'],baseline_error)       
        results_dictionary[f'baseline_rmse'] = np.append(results_dictionary[f'baseline_rmse'], baseline_rmse)
        results_dictionary[f'baseline_mape'] = np.append(results_dictionary[f'baseline_mape'], baseline_forecast)

        results_dictionary[f'simple_exponential_smoothing_forecast'] = np.append(results_dictionary[f'simple_exponential_smoothing_forecast'], simple_exponential_smoothing_forecast)
        results_dictionary[f'simple_exponential_smoothing_error'] = np.append(results_dictionary[f'simple_exponential_smoothing_error'], simple_exponential_smoothing_error)
        results_dictionary[f'simple_exponential_smoothing_rmse'] = np.append(results_dictionary[f'simple_exponential_smoothing_rmse'], simple_exponential_smoothing_rmse)
        results_dictionary[f'simple_exponential_smoothing_mape'] = np.append(results_dictionary[f'simple_exponential_smoothing_mape'], simple_exponential_smoothing_mape)

        results_dictionary[f'Linear_Regression_Forecast'] = np.append(results_dictionary[f'Linear_Regression_Forecast'], Linear_Regression_Forecast)
        results_dictionary[f'Linear_Regression_Error'] = np.append(results_dictionary[f'Linear_Regression_Error'], Linear_Regression_Error)
        results_dictionary[f'Linear_Regression_rmse'] = np.append(results_dictionary[f'Linear_Regression_rmse'], Linear_Regression_rmse)
        results_dictionary[f'Linear_Regression_mape'] = np.append(results_dictionary[f'Linear_Regression_mape'], Linear_Regression_mape)

        results_dictionary[f'holt_add_forecast'] = np.append(results_dictionary[f'holt_add_forecast'], holt_add_forecast)
        results_dictionary[f'holt_add_error'] = np.append(results_dictionary[f'holt_add_error'], holt_add_error)
        results_dictionary[f'holt_add_rmse'] = np.append(results_dictionary[f'holt_add_rmse'], holt_add_rmse)
        results_dictionary[f'holt_add_mape'] = np.append(results_dictionary[f'holt_add_mape'], holt_add_mape)

        results_dictionary[f'holt_mul_forecast'] = np.append(results_dictionary[f'holt_mul_forecast'], holt_mul_forecast)
        results_dictionary[f'holt_mul_error'] = np.append(results_dictionary[f'holt_mul_error'], holt_mul_error)
        results_dictionary[f'holt_mul_rmse'] = np.append(results_dictionary[f'holt_mul_rmse'], holt_mul_rmse)
        results_dictionary[f'holt_mul_mape'] = np.append(results_dictionary[f'holt_mul_mape'], holt_mul_mape)

        results_dictionary[f'holt_winters_add_forecast'] = np.append(results_dictionary[f'holt_winters_add_forecast'], holt_winters_add_forecast)
        results_dictionary[f'holt_winters_add_error'] = np.append(results_dictionary[f'holt_winters_add_error'], holt_winters_add_error)
        results_dictionary[f'holt_winters_add_rmse'] = np.append(results_dictionary[f'holt_winters_add_rmse'], holt_winters_add_rmse)
        results_dictionary[f'holt_winters_add_mape'] = np.append(results_dictionary[f'holt_winters_add_mape'], holt_winters_add_mape)

        results_dictionary[f'holt_winters_mult_forecast'] = np.append(results_dictionary[f'holt_winters_mult_forecast'], holt_winters_mult_forecast)
        results_dictionary[f'holt_winters_mult_error'] = np.append(results_dictionary[f'holt_winters_mult_error'], holt_winters_mult_error)
        results_dictionary[f'holt_winters_mult_rmse'] = np.append(results_dictionary[f'holt_winters_mult_rmse'], holt_winters_mult_rmse)
        results_dictionary[f'holt_winters_mult_mape'] = np.append(results_dictionary[f'holt_winters_mult_mape'], holt_winters_mult_mape)



    iterations += 1


for product_name in value_low_products:
    dataset_name = product_name
    serie = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    
    product_array = [product_name for i in range(3)]
    value_type = ['value_low' for i in range(3)]
    real_values = test_base


    baseline = random_walk(data=train_base.values, step=1)
    baseline_forecast = baseline.predict()
    baseline_error = real_values - baseline_forecast
    baseline_rmse = [root_mean_squared_error(real_values, baseline_forecast) for i in range(3)]
    baseline_mape = [mean_absolute_percentage_error(real_values, baseline_forecast) for i in range(3)]

    models = UFPR_Models(train_base=train_base.values.reshape(len(train_base.values)))
    models.Simple_Exponential_Smoothing()
    simple_exponential_smoothing_forecast = models.ses_forecast
    simple_exponential_smoothing_error = real_values - simple_exponential_smoothing_forecast
    simple_exponential_smoothing_rmse = [root_mean_squared_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]
    simple_exponential_smoothing_mape = [mean_absolute_percentage_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]

    models.Linear_Regression()
    Linear_Regression_Forecast = models.Linear_Regression_Forecast
    Linear_Regression_Error = real_values - Linear_Regression_Forecast
    Linear_Regression_rmse = [root_mean_squared_error(real_values, Linear_Regression_Forecast) for i in range(3)]
    Linear_Regression_mape = [mean_absolute_percentage_error(real_values, Linear_Regression_Forecast) for i in range(3)]

    trend = 'add'
    models.Holt_Exponential_Smoothing(trend)
    holt_add_forecast =  models.holt_forecast
    holt_add_error = real_values - holt_add_forecast
    holt_add_rmse = [root_mean_squared_error(real_values, holt_add_forecast) for i in range(3)]
    holt_add_mape = [mean_absolute_percentage_error(real_values, holt_add_forecast) for i in range(3)]

    trend = 'mul'
    models.Holt_Exponential_Smoothing(trend)
    holt_mul_forecast =  models.holt_forecast
    holt_mul_error =  real_values - holt_mul_forecast
    holt_mul_rmse = [root_mean_squared_error(real_values, holt_mul_forecast) for i in range(3)]
    holt_mul_mape = [mean_absolute_percentage_error(real_values, holt_mul_forecast) for i in range(3)]

    decomposition_type = 'add'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_add_forecast = models.holt_winters_forecast
    holt_winters_add_error = real_values - holt_winters_add_forecast
    holt_winters_add_rmse =[root_mean_squared_error(real_values, holt_winters_add_forecast) for i in range(3)]
    holt_winters_add_mape = [mean_absolute_percentage_error(real_values, holt_winters_add_forecast) for i in range(3)]

    decomposition_type = 'mul'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_mult_forecast = models.holt_winters_forecast
    holt_winters_mult_error = real_values - holt_winters_mult_forecast
    holt_winters_mult_rmse = [root_mean_squared_error(real_values, holt_winters_mult_forecast) for i in range(3)]
    holt_winters_mult_mape = [mean_absolute_percentage_error(real_values, holt_winters_mult_forecast) for i in range(3)]



    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param
    arima = UFPR_Models(train_base=train_base.values)

    for p in p_value:
        for d in d_value:
            for q in q_value:
                params = (p,d,q)
                
                try:
                    arima.Evaluate_Arima(params)
                    arima_predictions = arima.predictions[0].reshape(len(arima.predictions[0]))

                    arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],arima_predictions)
                    arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'], (real_values - arima_predictions))
                    arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], [root_mean_squared_error(real_values, arima_predictions) for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],[mean_absolute_percentage_error(real_values, arima_predictions) for i in range(3)])
                except:
                    print('LU_decomposition error')
                    arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'],['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], ['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],['decomposition error' for i in range(3)])
    

    results_dictionary[f'CODE_DISPLAY_NAME'] =np.append(results_dictionary[f'CODE_DISPLAY_NAME'], product_array)

    results_dictionary[f'real_values'] = np.append(results_dictionary[f'real_values'], real_values)
    results_dictionary[f'value_type_description'] = np.append(results_dictionary[f'value_type_description'], value_type)
    results_dictionary[f'Month_end'] = np.append(results_dictionary[f'Month_end'], Month_end)
    results_dictionary[f'baseline_forecast'] = np.append(results_dictionary[f'baseline_forecast'],baseline_forecast)
    results_dictionary[f'baseline_error'] = np.append(results_dictionary[f'baseline_error'],baseline_error)       
    results_dictionary[f'baseline_rmse'] = np.append(results_dictionary[f'baseline_rmse'], baseline_rmse)
    results_dictionary[f'baseline_mape'] = np.append(results_dictionary[f'baseline_mape'], baseline_forecast)
    results_dictionary[f'simple_exponential_smoothing_forecast'] = np.append(results_dictionary[f'simple_exponential_smoothing_forecast'], simple_exponential_smoothing_forecast)
    results_dictionary[f'simple_exponential_smoothing_error'] = np.append(results_dictionary[f'simple_exponential_smoothing_error'], simple_exponential_smoothing_error)
    results_dictionary[f'simple_exponential_smoothing_rmse'] = np.append(results_dictionary[f'simple_exponential_smoothing_rmse'], simple_exponential_smoothing_rmse)
    results_dictionary[f'simple_exponential_smoothing_mape'] = np.append(results_dictionary[f'simple_exponential_smoothing_mape'], simple_exponential_smoothing_mape)
    results_dictionary[f'Linear_Regression_Forecast'] = np.append(results_dictionary[f'Linear_Regression_Forecast'], Linear_Regression_Forecast)
    results_dictionary[f'Linear_Regression_Error'] = np.append(results_dictionary[f'Linear_Regression_Error'], Linear_Regression_Error)
    results_dictionary[f'Linear_Regression_rmse'] = np.append(results_dictionary[f'Linear_Regression_rmse'], Linear_Regression_rmse)
    results_dictionary[f'Linear_Regression_mape'] = np.append(results_dictionary[f'Linear_Regression_mape'], Linear_Regression_mape)
    results_dictionary[f'holt_add_forecast'] = np.append(results_dictionary[f'holt_add_forecast'], holt_add_forecast)
    results_dictionary[f'holt_add_error'] = np.append(results_dictionary[f'holt_add_error'], holt_add_error)
    results_dictionary[f'holt_add_rmse'] = np.append(results_dictionary[f'holt_add_rmse'], holt_add_rmse)
    results_dictionary[f'holt_add_mape'] = np.append(results_dictionary[f'holt_add_mape'], holt_add_mape)
    results_dictionary[f'holt_mul_forecast'] = np.append(results_dictionary[f'holt_mul_forecast'], holt_mul_forecast)
    results_dictionary[f'holt_mul_error'] = np.append(results_dictionary[f'holt_mul_error'], holt_mul_error)
    results_dictionary[f'holt_mul_rmse'] = np.append(results_dictionary[f'holt_mul_rmse'], holt_mul_rmse)
    results_dictionary[f'holt_mul_mape'] = np.append(results_dictionary[f'holt_mul_mape'], holt_mul_mape)
    results_dictionary[f'holt_winters_add_forecast'] = np.append(results_dictionary[f'holt_winters_add_forecast'], holt_winters_add_forecast)
    results_dictionary[f'holt_winters_add_error'] = np.append(results_dictionary[f'holt_winters_add_error'], holt_winters_add_error)
    results_dictionary[f'holt_winters_add_rmse'] = np.append(results_dictionary[f'holt_winters_add_rmse'], holt_winters_add_rmse)
    results_dictionary[f'holt_winters_add_mape'] = np.append(results_dictionary[f'holt_winters_add_mape'], holt_winters_add_mape)
    results_dictionary[f'holt_winters_mult_forecast'] = np.append(results_dictionary[f'holt_winters_mult_forecast'], holt_winters_mult_forecast)
    results_dictionary[f'holt_winters_mult_error'] = np.append(results_dictionary[f'holt_winters_mult_error'], holt_winters_mult_error)
    results_dictionary[f'holt_winters_mult_rmse'] = np.append(results_dictionary[f'holt_winters_mult_rmse'], holt_winters_mult_rmse)
    results_dictionary[f'holt_winters_mult_mape'] = np.append(results_dictionary[f'holt_winters_mult_mape'], holt_winters_mult_mape)


    





for product_name in midpoint_products:
    dataset_name = product_name
    serie = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    product_array = [product_name for i in range(3)]
    value_type = ['midpoint' for i in range(3)]
    real_values = test_base

    baseline = random_walk(data=train_base.values, step=1)
    baseline_forecast = baseline.predict()
    baseline_error = real_values - baseline_forecast
    baseline_rmse = [root_mean_squared_error(real_values, baseline_forecast) for i in range(3)]
    baseline_mape = [mean_absolute_percentage_error(real_values, baseline_forecast) for i in range(3)]

    models = UFPR_Models(train_base=train_base.values.reshape(len(train_base.values)))
    models.Simple_Exponential_Smoothing()
    simple_exponential_smoothing_forecast = models.ses_forecast
    simple_exponential_smoothing_error = real_values - simple_exponential_smoothing_forecast
    simple_exponential_smoothing_rmse = [root_mean_squared_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]
    simple_exponential_smoothing_mape = [mean_absolute_percentage_error(real_values, simple_exponential_smoothing_forecast) for i in range(3)]

    models.Linear_Regression()
    Linear_Regression_Forecast = models.Linear_Regression_Forecast
    Linear_Regression_Error = real_values - Linear_Regression_Forecast
    Linear_Regression_rmse = [root_mean_squared_error(real_values, Linear_Regression_Forecast) for i in range(3)]
    Linear_Regression_mape = [mean_absolute_percentage_error(real_values, Linear_Regression_Forecast) for i in range(3)]

    trend = 'add'
    models.Holt_Exponential_Smoothing(trend)
    holt_add_forecast =  models.holt_forecast
    holt_add_error = real_values - holt_add_forecast
    holt_add_rmse = [root_mean_squared_error(real_values, holt_add_forecast) for i in range(3)]
    holt_add_mape = [mean_absolute_percentage_error(real_values, holt_add_forecast) for i in range(3)]

    trend = 'mul'
    models.Holt_Exponential_Smoothing(trend)
    holt_mul_forecast =  models.holt_forecast
    holt_mul_error =  real_values - holt_mul_forecast
    holt_mul_rmse = [root_mean_squared_error(real_values, holt_mul_forecast) for i in range(3)]
    holt_mul_mape = [mean_absolute_percentage_error(real_values, holt_mul_forecast) for i in range(3)]

    decomposition_type = 'add'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_add_forecast = models.holt_winters_forecast
    holt_winters_add_error = real_values - holt_winters_add_forecast
    holt_winters_add_rmse =[root_mean_squared_error(real_values, holt_winters_add_forecast) for i in range(3)]
    holt_winters_add_mape = [mean_absolute_percentage_error(real_values, holt_winters_add_forecast) for i in range(3)]

    decomposition_type = 'mul'
    models.Holt_Winters(decomposition_type, freq=12)
    holt_winters_mult_forecast = models.holt_winters_forecast
    holt_winters_mult_error = real_values - holt_winters_mult_forecast
    holt_winters_mult_rmse = [root_mean_squared_error(real_values, holt_winters_mult_forecast) for i in range(3)]
    holt_winters_mult_mape = [mean_absolute_percentage_error(real_values, holt_winters_mult_forecast) for i in range(3)]



    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param
    arima = UFPR_Models(train_base=train_base.values)

    for p in p_value:
        for d in d_value:
            for q in q_value:
                params = (p,d,q)
                try:
                    arima.Evaluate_Arima(params)
                    arima_predictions = arima.predictions[0].reshape(len(arima.predictions[0]))

                    arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],arima_predictions)
                    arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'], (real_values - arima_predictions))
                    arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], [root_mean_squared_error(real_values, arima_predictions) for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],[mean_absolute_percentage_error(real_values, arima_predictions) for i in range(3)])
                except:
                    print('LU_decomposition error')
                    arima_dictionary[f'arima({p},{d},{q}) forecast'] = np.append(arima_dictionary[f'arima({p},{d},{q}) forecast'],['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) error'] = np.append(arima_dictionary[f'arima({p},{d},{q}) error'],['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) rmse'] = np.append(arima_dictionary[f'arima({p},{d},{q}) rmse'], ['decomposition error' for i in range(3)])
                    arima_dictionary[f'arima({p},{d},{q}) mape'] = np.append(arima_dictionary[f'arima({p},{d},{q}) mape'],['decomposition error' for i in range(3)])
    
    results_dictionary[f'CODE_DISPLAY_NAME'] =np.append(results_dictionary[f'CODE_DISPLAY_NAME'], product_array)

    results_dictionary[f'value_type_description'] = np.append(results_dictionary[f'value_type_description'], value_type)
    results_dictionary[f'Month_end'] = np.append(results_dictionary[f'Month_end'], Month_end)
    results_dictionary[f'real_values'] = np.append(results_dictionary[f'real_values'], real_values)
    results_dictionary[f'baseline_forecast'] = np.append(results_dictionary[f'baseline_forecast'],baseline_forecast)
    results_dictionary[f'baseline_error'] = np.append(results_dictionary[f'baseline_error'],baseline_error)       
    results_dictionary[f'baseline_rmse'] = np.append(results_dictionary[f'baseline_rmse'], baseline_rmse)
    results_dictionary[f'baseline_mape'] = np.append(results_dictionary[f'baseline_mape'], baseline_forecast)
    results_dictionary[f'simple_exponential_smoothing_forecast'] = np.append(results_dictionary[f'simple_exponential_smoothing_forecast'], simple_exponential_smoothing_forecast)
    results_dictionary[f'simple_exponential_smoothing_error'] = np.append(results_dictionary[f'simple_exponential_smoothing_error'], simple_exponential_smoothing_error)
    results_dictionary[f'simple_exponential_smoothing_rmse'] = np.append(results_dictionary[f'simple_exponential_smoothing_rmse'], simple_exponential_smoothing_rmse)
    results_dictionary[f'simple_exponential_smoothing_mape'] = np.append(results_dictionary[f'simple_exponential_smoothing_mape'], simple_exponential_smoothing_mape)
    results_dictionary[f'Linear_Regression_Forecast'] = np.append(results_dictionary[f'Linear_Regression_Forecast'], Linear_Regression_Forecast)
    results_dictionary[f'Linear_Regression_Error'] = np.append(results_dictionary[f'Linear_Regression_Error'], Linear_Regression_Error)
    results_dictionary[f'Linear_Regression_rmse'] = np.append(results_dictionary[f'Linear_Regression_rmse'], Linear_Regression_rmse)
    results_dictionary[f'Linear_Regression_mape'] = np.append(results_dictionary[f'Linear_Regression_mape'], Linear_Regression_mape)
    results_dictionary[f'holt_add_forecast'] = np.append(results_dictionary[f'holt_add_forecast'], holt_add_forecast)
    results_dictionary[f'holt_add_error'] = np.append(results_dictionary[f'holt_add_error'], holt_add_error)
    results_dictionary[f'holt_add_rmse'] = np.append(results_dictionary[f'holt_add_rmse'], holt_add_rmse)
    results_dictionary[f'holt_add_mape'] = np.append(results_dictionary[f'holt_add_mape'], holt_add_mape)
    results_dictionary[f'holt_mul_forecast'] = np.append(results_dictionary[f'holt_mul_forecast'], holt_mul_forecast)
    results_dictionary[f'holt_mul_error'] = np.append(results_dictionary[f'holt_mul_error'], holt_mul_error)
    results_dictionary[f'holt_mul_rmse'] = np.append(results_dictionary[f'holt_mul_rmse'], holt_mul_rmse)
    results_dictionary[f'holt_mul_mape'] = np.append(results_dictionary[f'holt_mul_mape'], holt_mul_mape)
    results_dictionary[f'holt_winters_add_forecast'] = np.append(results_dictionary[f'holt_winters_add_forecast'], holt_winters_add_forecast)
    results_dictionary[f'holt_winters_add_error'] = np.append(results_dictionary[f'holt_winters_add_error'], holt_winters_add_error)
    results_dictionary[f'holt_winters_add_rmse'] = np.append(results_dictionary[f'holt_winters_add_rmse'], holt_winters_add_rmse)
    results_dictionary[f'holt_winters_add_mape'] = np.append(results_dictionary[f'holt_winters_add_mape'], holt_winters_add_mape)
    results_dictionary[f'holt_winters_mult_forecast'] = np.append(results_dictionary[f'holt_winters_mult_forecast'], holt_winters_mult_forecast)
    results_dictionary[f'holt_winters_mult_error'] = np.append(results_dictionary[f'holt_winters_mult_error'], holt_winters_mult_error)
    results_dictionary[f'holt_winters_mult_rmse'] = np.append(results_dictionary[f'holt_winters_mult_rmse'], holt_winters_mult_rmse)
    results_dictionary[f'holt_winters_mult_mape'] = np.append(results_dictionary[f'holt_winters_mult_mape'], holt_winters_mult_mape)

 
    


results_dictionary.update(arima_dictionary)
df = pd.DataFrame(results_dictionary)

filepath = r'C:\Users\User\Desktop\Exxon\git akhub\akhub\JULIO\scripts\Data_Pre_Processing\Cleaned_Datasets\vertical_forecasts2.csv'
df.to_csv(filepath)

# %%

###################################################################
####################### REDES NEURAIS #############################



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Classical_Statistics_Models.classical_statistics_models import UFPR_Models
from Machine_Learning_Models.RBF_model import rbf_model
from Machine_Learning_Models.kmeans_RBF import kmeans_rbf
from Classical_Statistics_Models.Baseline import random_walk

dataset_path =  r'C:\Users\User\Desktop\Exxon\Forecasting ARGUS base\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\scripts\Data_Pre_Processing\Cleaned_Datasets\argus_clean.csv'
dataset = pd.read_csv(dataset_path)



train_window_size = 48
test_window_size = 3

gamma = 0.8
n_input = 3
n_clusters = 10

input_dim = 3
output_dim = 1
units= 10
batch_size=48
epochs = 3000
## name of the base oils

value_low_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value low'].copy()
value_high_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value high'].copy()
midpoint_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'midpoint'].copy()

products = list(dataset['CODE_DISPLAY_NAME'].unique())
value_high_products = list(value_high_df['CODE_DISPLAY_NAME'].unique())
value_low_products = list(value_low_df['CODE_DISPLAY_NAME'].unique())
midpoint_products = list(midpoint_df['CODE_DISPLAY_NAME'].unique())
print(len(value_high_products))

for product_name in value_high_products:
    iterations = 1
    exception = False
    dataset_name = product_name
    serie = value_high_df[value_high_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    product_array = [product_name for i in range(3)]
    value_type = ['value_high' for i in range(3)]
    real_values = test_base

    ## K-means RBF
    models = kmeans_rbf(serie=train_base.reshape(len(train_base.values)), n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
    models.predictive_predictor_split(n_input)
    models.scale()
    try:
        models.model_fit()
    except:
        print("Singular Matrix found")
        exception = True
        continue
    rbf_kmeans_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    if exception == False:
        for i in range(n_input, 0, -1):
            real_values = train_base.values[(-i):].reshape(len(train_base.values))
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_kmeans_predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict =  train_base.values[(-i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            rbf_kmeans_predicts.append(predict)
            predict_iterations += 1
            if predict_iterations == 3:
                break
    if exception == True:
        rbf_kmeans_predicts = ['singular matrix found (failed)' for i in range(3)]

    ## Grasp RBF

    models = rbf_model(serie=train_base.reshape(len(train_base.values)), units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
    models.predictive_predictor_split()
    models.scale()
    models.fit_model(epochs=epochs, batch_size=batch_size)

    rbf_grasp_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    for i in range(n_input, 0, -1):
        real_values = train_base.values[(-i):].reshape(len(train_base.values))
        # print(f'Real_Values: {real_values}')
        if i < n_input:
            input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_grasp_predicts).flatten()).reshape(-1,n_input)
        else: 
            input_predict =  train_base.values[(-i):].reshape(-1,n_input)
        # print(f'input_predict={input_predict}')
        scaled_pred = models.predict(input_predict)
        predict = models.inverse_scale(scaled_pred)
        rbf_grasp_predicts.append(predict)
        predict_iterations += 1
        if predict_iterations == 3:
            break


    if iterations == 1:
        results_dictionary['rbf_kmeans_predicts'] = rbf_kmeans_predicts
        results_dictionary['rbf_kmeans_error'] = test_base - rbf_kmeans_predicts
        results_dictionary['rbf_kmeans_rmse'] =  [root_mean_squared_error(rbf_kmeans_predicts, test_base) for i in range(3)]
        results_dictionary['rbf_kmeans_mape'] = [mean_absolute_percentage_error(rbf_kmeans_predicts, test_base) for i in range(3)]

        results_dictionary['rbf_grasp_predicts'] = rbf_grasp_predicts
        results_dictionary['rbf_grasp_error'] = test_base - rbf_grasp_predicts
        results_dictionary['rbf_grasp_rmse'] =  [root_mean_squared_error(rbf_grasp_predicts, test_base) for i in range(3)]
        results_dictionary['rbf_grasp_mape'] = [mean_absolute_percentage_error(rbf_grasp_predicts, test_base) for i in range(3)]
    else:
        results_dictionary['rbf_kmeans_predicts'] = np.append(results_dictionary['rbf_kmeans_predicts'], rbf_kmeans_predicts)
        results_dictionary['rbf_kmeans_error'] = np.append(results_dictionary['rbf_kmeans_error'], test_base - rbf_kmeans_predicts)
        results_dictionary['rbf_kmeans_rmse'] =  np.append(results_dictionary['rbf_kmeans_rmse'], [root_mean_squared_error(rbf_kmeans_predicts, test_base) for i in range(3)])
        results_dictionary['rbf_kmeans_mape'] =  np.append(results_dictionary['rbf_kmeans_mape'], [mean_absolute_percentage_error(rbf_kmeans_predicts, test_base) for i in range(3)])
   
        results_dictionary['rbf_grasp_predicts'] = np.append(results_dictionary['rbf_grasp_predicts'], rbf_grasp_predicts)
        results_dictionary['rbf_grasp_error'] = np.append(results_dictionary['rbf_grasp_error'], test_base - rbf_grasp_predicts)
        results_dictionary['rbf_grasp_rmse'] =  np.append(results_dictionary['rbf_grasp_rmse'], [root_mean_squared_error(rbf_grasp_predicts, test_base) for i in range(3)])
        results_dictionary['rbf_grasp_mape'] =  np.append(results_dictionary['rbf_grasp_mape'], [mean_absolute_percentage_error(rbf_grasp_predicts, test_base) for i in range(3)])
    iterations += 1

#########################

for product_name in value_low_products:
    exception = False
    dataset_name = product_name
    serie = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    product_array = [product_name for i in range(3)]
    value_type = ['value_low' for i in range(3)]
    real_values = test_base

    ## K-means RBF
    models = kmeans_rbf(serie=train_base.reshape(len(train_base.values)), n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
    models.predictive_predictor_split(n_input)
    models.scale()
    try:
        models.model_fit()
    except:
        print("Singular Matrix found")
        exception = True
        continue
    rbf_kmeans_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    if exception == False:
        for i in range(n_input, 0, -1):
            real_values = train_base.values[(-i):].reshape(len(train_base.values))
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_kmeans_predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict =  train_base.values[(-i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            rbf_kmeans_predicts.append(predict)
            predict_iterations += 1
            if predict_iterations == 3:
                break
    if exception == True:
        rbf_kmeans_predicts = ['singular matrix found (failed)' for i in range(3)]

    ## Grasp RBF

    models = rbf_model(serie=train_base.reshape(len(train_base.values)), units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
    models.predictive_predictor_split()
    models.scale()
    models.fit_model(epochs=epochs, batch_size=batch_size)

    rbf_grasp_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    for i in range(n_input, 0, -1):
        real_values = train_base.values[(-i):].reshape(len(train_base.values))
        # print(f'Real_Values: {real_values}')
        if i < n_input:
            input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_grasp_predicts).flatten()).reshape(-1,n_input)
        else: 
            input_predict =  train_base.values[(-i):].reshape(-1,n_input)
        # print(f'input_predict={input_predict}')
        scaled_pred = models.predict(input_predict)
        predict = models.inverse_scale(scaled_pred)
        rbf_grasp_predicts.append(predict)
        predict_iterations += 1
        if predict_iterations == 3:
            break



    results_dictionary['rbf_kmeans_predicts'] = np.append(results_dictionary['rbf_kmeans_predicts'], rbf_kmeans_predicts)
    results_dictionary['rbf_kmeans_error'] = np.append(results_dictionary['rbf_kmeans_error'], test_base - rbf_kmeans_predicts)
    results_dictionary['rbf_kmeans_rmse'] =  np.append(results_dictionary['rbf_kmeans_rmse'], [root_mean_squared_error(rbf_kmeans_predicts, test_base) for i in range(3)])
    results_dictionary['rbf_kmeans_mape'] =  np.append(results_dictionary['rbf_kmeans_mape'], [mean_absolute_percentage_error(rbf_kmeans_predicts, test_base) for i in range(3)])

    results_dictionary['rbf_grasp_predicts'] = np.append(results_dictionary['rbf_grasp_predicts'], rbf_grasp_predicts)
    results_dictionary['rbf_grasp_error'] = np.append(results_dictionary['rbf_grasp_error'], test_base - rbf_grasp_predicts)
    results_dictionary['rbf_grasp_rmse'] =  np.append(results_dictionary['rbf_grasp_rmse'], [root_mean_squared_error(rbf_grasp_predicts, test_base) for i in range(3)])
    results_dictionary['rbf_grasp_mape'] =  np.append(results_dictionary['rbf_grasp_mape'], [mean_absolute_percentage_error(rbf_grasp_predicts, test_base) for i in range(3)])


#############################


for product_name in midpoint_products:
    iterations = 1
    exception = False
    dataset_name = product_name
    serie = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    train_base = serie.iloc[-(48+3):-3]
    test_base = serie.iloc[-3:].values.reshape(len(serie.iloc[-3:].values))
    Month_end = serie.iloc[-3:].index.values.reshape(len(serie.iloc[-3:].index.values))
    product_array = [product_name for i in range(3)]
    value_type = ['midpoint' for i in range(3)]
    real_values = test_base

    ## K-means RBF
    models = kmeans_rbf(serie=train_base.reshape(len(train_base.values)), n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
    models.predictive_predictor_split(n_input)
    models.scale()
    try:
        models.model_fit()
    except:
        print("Singular Matrix found")
        exception = True
        continue
    rbf_kmeans_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    if exception == False:
        for i in range(n_input, 0, -1):
            real_values = train_base.values[(-i):].reshape(len(train_base.values))
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_kmeans_predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict =  train_base.values[(-i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            rbf_kmeans_predicts.append(predict)
            predict_iterations += 1
            if predict_iterations == 3:
                break
    if exception == True:
        rbf_kmeans_predicts = ['singular matrix found (failed)' for i in range(3)]

    ## Grasp RBF

    models = rbf_model(serie=train_base.reshape(len(train_base.values)), units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
    models.predictive_predictor_split()
    models.scale()
    models.fit_model(epochs=epochs, batch_size=batch_size)

    rbf_grasp_predicts = []
    ##saving the results on the predictions matrix
    predict_iterations = 0

    for i in range(n_input, 0, -1):
        real_values = train_base.values[(-i):].reshape(len(train_base.values))
        # print(f'Real_Values: {real_values}')
        if i < n_input:
            input_predict = np.append(np.array(real_values).flatten(),np.array(rbf_grasp_predicts).flatten()).reshape(-1,n_input)
        else: 
            input_predict =  train_base.values[(-i):].reshape(-1,n_input)
        # print(f'input_predict={input_predict}')
        scaled_pred = models.predict(input_predict)
        predict = models.inverse_scale(scaled_pred)
        rbf_grasp_predicts.append(predict)
        predict_iterations += 1
        if predict_iterations == 3:
            break



    results_dictionary['rbf_kmeans_predicts'] = np.append(results_dictionary['rbf_kmeans_predicts'], rbf_kmeans_predicts)
    results_dictionary['rbf_kmeans_error'] = np.append(results_dictionary['rbf_kmeans_error'], test_base - rbf_kmeans_predicts)
    results_dictionary['rbf_kmeans_rmse'] =  np.append(results_dictionary['rbf_kmeans_rmse'], [root_mean_squared_error(rbf_kmeans_predicts, test_base) for i in range(3)])
    results_dictionary['rbf_kmeans_mape'] =  np.append(results_dictionary['rbf_kmeans_mape'], [mean_absolute_percentage_error(rbf_kmeans_predicts, test_base) for i in range(3)])

    results_dictionary['rbf_grasp_predicts'] = np.append(results_dictionary['rbf_grasp_predicts'], rbf_grasp_predicts)
    results_dictionary['rbf_grasp_error'] = np.append(results_dictionary['rbf_grasp_error'], test_base - rbf_grasp_predicts)
    results_dictionary['rbf_grasp_rmse'] =  np.append(results_dictionary['rbf_grasp_rmse'], [root_mean_squared_error(rbf_grasp_predicts, test_base) for i in range(3)])
    results_dictionary['rbf_grasp_mape'] =  np.append(results_dictionary['rbf_grasp_mape'], [mean_absolute_percentage_error(rbf_grasp_predicts, test_base) for i in range(3)])

df = pd.DataFrame(results_dictionary)

filepath = r'C:\Users\User\Desktop\Exxon\git akhub\akhub\JULIO\scripts\Data_Pre_Processing\Cleaned_Datasets\Neural_Networks_forecasts.csv'
df.to_csv(filepath)
# %%
