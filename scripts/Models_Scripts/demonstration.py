#%%
import numpy as np
import pandas as pd
## Preparing time windows and enviroment to receive the forecasts
from utility import TimeSeries
from models import UFPR_Models

#%%



'''
dataset_path: variable that contains the path of the dataset  
dataset_name: this will be the name displayed on the excel file for this dataset
save_path: path where excel file with the results will be saved
freq: frequency of the observations: i.e. Weekly, Daily, Monthly
'''


dataset_path =  '../Data_Pre_Processing/treated_datasets/new_argus_series.xlsx'
save_path= '../../closed_datasets/Closed_results/TESTS_RESULTS.csv'
freq = 'Monthly'
value_description = 'pricing'
## Calling the dataset
dataset = pd.read_csv(dataset_path)

columns = dataset.columns


#%%
## Setting dataframes with date and values of the series




## First, we want to forecast midpoint with monthly frequency
dataset = dataset[dataset.value_description == 'midpoint']
dataset = dataset[dataset.Frequency == 'M']

## name of the base oils
products = dataset['CODE_NAME'].unique()
products




#%%
## Filling NaN values and dropping series with less than the minimum size (train_window_size + test_window_size):
fillnamethod = 'interpolate'
for product in products:
    dataset[dataset.CODE_NAME == product] = dataset[dataset.CODE_NAME == product].interpolate()
    if len(dataset[dataset.CODE_NAME == product].VALUE.values) < train_window_size + test_window_size:
        product.remove(product)
        
#%%
dataset
#%%
#%%
dataset[dataset.CODE_NAME == products[0]]


#%%
## Filling NaN values:
fillnamethod = 'interpolate'
for product in products:
    dataset[dataset.CODE_NAME == product] = dataset[dataset.CODE_NAME == product].interpolate()



#%%

'''
MAKING ONE-CODE CALL FOR ALL MODELS
'''

#%% 

df = pd.read_csv(r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast\Open_datasets\Crude_Oil.csv')
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)
series = df.resample('M').agg('mean')
series.head()

#%%


for product_name in products:
    dataset_name = product_name
    serie = dataset[dataset.CODE_NAME == product_name]
    serie = serie[['OPR_DATE', 'VALUE']]
    serie.set_index('OPR_DATE', inplace=True)
    time_series = TimeSeries(name='crude_oil', series=series.value, train_window_size=24, test_window_size=3, frequency=freq, fillna=fillnamethod, value_description=value_description)
    time_series.create_windows()


    '''
    SIMPLE EXPONENTIAL SMOOTHING 
    '''
    
    ## Calling Simple Exponential Smoothing for every window and saving the results on predictions matrix
    for j in range(time_series.n_predictions):
        ##calling the model
        models = UFPR_Models(train_base=time_series.train_windows[j])
        models.Simple_Exponential_Smoothing()
        ##saving the results on the predictions matrix
        time_series.predictions[j] = models.ses_forecast
    ## calculating scores
    time_series.evaluate_model()
    ## saving results
    time_series.save_data(model=f'Simple Exponential Smoothing',filepath=save_path)
    
    

    '''
    LINEAR REGRESSION 
    '''


    ## Calling Linear Regression for every window and saving the results on predictions matrix
    for j in range(time_series.n_predictions):
        ##calling the model
        models = UFPR_Models(train_base=time_series.train_windows[j])
        models.Linear_Regression()
        ##saving the results on the predictions matrix
        time_series.predictions[j] = models.Linear_Regression_Forecast
    ## calculating the scores and saving results on an excel file
    time_series.evaluate_model()
    time_series.save_data(model=f'Linear Regression', filepath=save_path)
    

    '''
    Holt Exponential Smoothing model
    '''




    trend=['add','mul']
    for value in trend:
        ## Calling Simple Exponential Smoothing for every window and saving the results on predictions matrix
        for j in range(time_series.n_predictions):
            models = UFPR_Models(train_base=time_series.train_windows[j])
            ##calling the model
            models.Holt_Exponential_Smoothing(value)
            ##saving the results on the predictions matrix
            time_series.predictions[j] = models.holt_forecast
        ## calculating the scores and saving results on an excel file
        time_series.evaluate_model()
        time_series.save_data(model=f'Holt_Exponential_Smoothing - {value}',filepath=save_path)



    '''
    Holt Winters model
    '''

    import numpy as np
    import pandas as pd


    seasonal=['add', 'mul']
    ## Calling Holt Winters model for every window and saving the results on predictions matrix
    for decomposition_type in seasonal:
        for j in range(time_series.n_predictions):
            df = pd.Series(data=time_series.train_windows[j], index=time_series.date_index[j:(j+time_series.train_window_size)])
            ##calling the model
            models = UFPR_Models(train_base=df.values, train_base_index=df.index)
            models.Holt_Winters(decomposition_type, freq=12)
            ##saving the results on the predictions matrix
            time_series.predictions[j] = models.holt_winters_forecast

        ## calculating the scores and saving results on an excel file
        time_series.evaluate_model()
        time_series.save_data(model=f'Holt_Winters - {decomposition_type}',filepath=save_path)



    ## ARIMA

    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]

    ## for every grid param

    for p in p_value:
        for d in d_value:
            for q in q_value:


                for j in range(time_series.n_predictions):
                    ##calling the model
                    arima = UFPR_Models(train_base=time_series.train_windows[j], evaluate_params=(p,d,q))
                    try: 
                        arima.Evaluate_Arima()
                        time_series.predictions[j] = arima.predictions[0]
                    except:
                        print('LU_decomposition error')
                        
                    ##saving the results on the predictions matrix
                    
                ##calculating scores
                time_series.evaluate_model()
                ##saving results
                time_series.save_data(model=f'Arima({p};{d};{q})',filepath=save_path)

    break


#%%

