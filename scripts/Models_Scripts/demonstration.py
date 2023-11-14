#%%
import numpy as np
import pandas as pd
from utility_copy import TimeSeries
from Classical_Statistics_Models.classical_statistics_models import UFPR_Models
from Machine_Learning_Models.RBF_model import rbf_model
from Machine_Learning_Models.kmeans_RBF import kmeans_rbf
from Classical_Statistics_Models.Baseline import random_walk
import warnings
import warnings

warnings.filterwarnings("ignore")

#%%
'''
dataset_path: variable that contains the path of the dataset  
dataset_name: this will be the name displayed on the excel file for this dataset
save_path: path where excel file with the results will be saved
freq: frequency of the observations: i.e. Weekly, Daily, Monthly
'''
warnings.filterwarnings("ignore")

dataset_path =  r'C:\Users\Work\Documents\GitHub\akhub\JULIO\scripts\Data_Pre_Processing\argus_clean.csv'
save_path= r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast - Copia\closed_datasets\Closed_results\TESTS_RESULTS_third_try.csv'
freq = 'Monthly'

## Calling the dataset
dataset = pd.read_csv(dataset_path)

columns = dataset.columns
## First, we want to forecast midpoint with monthly frequency

train_window_size = 48
test_window_size = 3
window_step = 1
## name of the base oils

value_low_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value low'].copy()
value_high_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'value high'].copy()
midpoint_df = dataset[dataset.PRICE_TYPE_DESCRIPTION == 'midpoint'].copy()

products = list(dataset['CODE_DISPLAY_NAME'].unique())
value_high_products = list(value_high_df['CODE_DISPLAY_NAME'].unique())
value_low_products = list(value_low_df['CODE_DISPLAY_NAME'].unique())
midpoint_products = list(midpoint_df['CODE_DISPLAY_NAME'].unique())
print(len(value_high_products))

## Filling NaN values and dropping series with less than the minimum size (train_window_size + test_window_size):
fillnamethod = 'interpolate'
for product in products:
    value_low_df[value_low_df.CODE_DISPLAY_NAME == product] = value_low_df[value_low_df.CODE_DISPLAY_NAME == product].interpolate()
    value_high_df[value_high_df.CODE_DISPLAY_NAME == product] = value_high_df[value_high_df.CODE_DISPLAY_NAME == product].interpolate()
    midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product] = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product].interpolate()
    if (product in value_low_df['CODE_DISPLAY_NAME'].values) and (len(value_low_df[value_low_df.CODE_DISPLAY_NAME == product].VALUE.values) <= train_window_size + test_window_size):
        value_low_products.remove(product)
    if (product in value_high_df['CODE_DISPLAY_NAME'].values) and (len(value_high_df[value_high_df.CODE_DISPLAY_NAME == product].VALUE.values) <= (train_window_size + test_window_size)):
        value_high_products.remove(product)
    if (product in midpoint_df['CODE_DISPLAY_NAME'].values) and (len(midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product].VALUE.values) <= train_window_size + test_window_size):
        midpoint_products.remove(product)
#%%
dataset
#%%
'''
MAKING ONE-CODE CALL FOR ALL MODELS
'''


for product_name in value_high_products:
    dataset_name = product_name
    serie = dataset[dataset.CODE_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    time_series = TimeSeries(name=dataset_name, series=dataset['value'], train_window_size=train_window_size, test_window_size=test_window_size, step=window_step, frequency=freq, fillna=fillnamethod, value_description = value_description)
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
    time_series.save_data(model=f'Simple Exponential Smoothing',PRICE_TYPE_DESCRIPTION='value_high' ,filepath=save_path)
    
    
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
    time_series.save_data(model=f'Linear Regression',PRICE_TYPE_DESCRIPTION='value_high', filepath=save_path)
    
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
        time_series.save_data(model=f'Holt_Exponential_Smoothing - {value}',PRICE_TYPE_DESCRIPTION='value_high',filepath=save_path)
    '''
    Holt Winters model
    '''

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
        time_series.save_data(model=f'Holt_Winters - {decomposition_type}',PRICE_TYPE_DESCRIPTION='value_high',filepath=save_path)
    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param

 
    for p in p_value:
        for d in d_value:
            for q in q_value:
                exception = False
                for j in range(time_series.n_predictions):   
                    ##calling the model
                    params = (p,d,q)
                    arima = UFPR_Models(train_base=time_series.train_windows[j].reshape(-1,1))

                    try: 
                        arima.Evaluate_Arima(params)
                        time_series.predictions[j] = arima.predictions[0]
                    except:
                        print('LU_decomposition error')
                        exception = True
                        
                    ##saving the results on the predictions matrix
                for k in range(len(time_series.predictions)):
                    if (time_series.predictions[k,0] == 0) and (time_series.predictions[k,1] == 0) and time_series.predictions[k,2] == 0:
                        delete_indexes.append(k)
                    delete_indexes = []
                time_series.predictions = np.delete(time_series.predictions,delete_indexes, axis=0)
                time_series.test_windows = np.delete(time_series.test_windows,delete_indexes, axis=0)
                time_series.evaluate_model()
                ##saving results
                time_series.save_data(model=f'Arima({p};{d};{q})',PRICE_TYPE_DESCRIPTION='value_high',filepath=save_path)



for product_name in value_low_products:
    dataset_name = product_name
    serie = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    time_series = TimeSeries(name=dataset_name, series=serie['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = 'value_low')

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
    time_series.save_data(model=f'Simple Exponential Smoothing',PRICE_TYPE_DESCRIPTION='value_low' ,filepath=save_path)
    
    
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
    time_series.save_data(model=f'Linear Regression',PRICE_TYPE_DESCRIPTION='value_low', filepath=save_path)
    
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
        time_series.save_data(model=f'Holt_Exponential_Smoothing - {value}',PRICE_TYPE_DESCRIPTION='value_low',filepath=save_path)
    '''
    Holt Winters model
    '''

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
        time_series.save_data(model=f'Holt_Winters - {decomposition_type}',PRICE_TYPE_DESCRIPTION='value_low',filepath=save_path)
    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param

 
    for p in p_value:
        for d in d_value:
            for q in q_value:
                exception = False
                for j in range(time_series.n_predictions):   
                    ##calling the model
                    params = (p,d,q)
                    arima = UFPR_Models(train_base=time_series.train_windows[j].reshape(-1,1))

                    try: 
                        arima.Evaluate_Arima(params)
                        time_series.predictions[j] = arima.predictions[0]
                    except:
                        print('LU_decomposition error')
                        exception = True
                        
                    ##saving the results on the predictions matrix
                for k in range(len(time_series.predictions)):
                    if (time_series.predictions[k,0] == 0) and (time_series.predictions[k,1] == 0) and time_series.predictions[k,2] == 0:
                        delete_indexes.append(k)
                    delete_indexes = []
                time_series.predictions = np.delete(time_series.predictions,delete_indexes, axis=0)
                time_series.test_windows = np.delete(time_series.test_windows,delete_indexes, axis=0)
                time_series.evaluate_model()
                ##saving results
                time_series.save_data(model=f'Arima({p};{d};{q})',PRICE_TYPE_DESCRIPTION='value_low',filepath=save_path)




for product_name in midpoint_products:
    dataset_name = product_name
    serie = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    serie = serie[['MONTH_END', 'VALUE']]
    serie.set_index('MONTH_END', inplace=True)
    time_series = TimeSeries(name=dataset_name, series=serie['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = 'midpoint')

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
    time_series.save_data(model=f'Simple Exponential Smoothing',PRICE_TYPE_DESCRIPTION='midpoint' ,filepath=save_path)
    
    
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
    time_series.save_data(model=f'Linear Regression',PRICE_TYPE_DESCRIPTION='midpoint', filepath=save_path)
    
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
        time_series.save_data(model=f'Holt_Exponential_Smoothing - {value}',PRICE_TYPE_DESCRIPTION='midpoint',filepath=save_path)
    '''
    Holt Winters model
    '''

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
        time_series.save_data(model=f'Holt_Winters - {decomposition_type}',PRICE_TYPE_DESCRIPTION='midpoint',filepath=save_path)
    ## ARIMA
    p_value = [0,1,2,3,4,5]
    d_value = [0,1,2,3,4,5,6]
    q_value = [0,1,2,3,4]
    ## for every grid param

 
    for p in p_value:
        for d in d_value:
            for q in q_value:
                exception = False
                for j in range(time_series.n_predictions):   
                    ##calling the model
                    params = (p,d,q)
                    arima = UFPR_Models(train_base=time_series.train_windows[j].reshape(-1,1))

                    try: 
                        arima.Evaluate_Arima(params)
                        time_series.predictions[j] = arima.predictions[0]
                    except:
                        print('LU_decomposition error')
                        exception = True
                        
                    ##saving the results on the predictions matrix
                for k in range(len(time_series.predictions)):
                    if (time_series.predictions[k,0] == 0) and (time_series.predictions[k,1] == 0) and time_series.predictions[k,2] == 0:
                        delete_indexes.append(k)
                    delete_indexes = []
                time_series.predictions = np.delete(time_series.predictions,delete_indexes, axis=0)
                time_series.test_windows = np.delete(time_series.test_windows,delete_indexes, axis=0)
                time_series.evaluate_model()
                ##saving results
                time_series.save_data(model=f'Arima({p};{d};{q})',PRICE_TYPE_DESCRIPTION='midpoint',filepath=save_path)

#%%
'''
RBF
'''


# df = pd.read_csv(dataset_path)
# df.rename(columns={' Value': 'Value'}, inplace=True)
# df.set_index("Date",inplace=True)
# df.index = pd.to_datetime(df.index)
# df.dropna(inplace=True)
# df = df.resample('M').agg('mean')
value_description = 'M'


input_dim = 5
output_dim = 1
units= 5
gamma = 0.8
# train_size = 1
batch_size=60
epochs = 3000

for product_name in value_high_products:
        
    dataset_name = product_name
    evaluate_df = value_high_df[value_high_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, window_step=window_step, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    
    for j in range(product_time_series.n_predictions):
        
        ##calling the model
        
        models = rbf_model(product_time_series.train_windows[j], units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
        models.scale()
        models.predictive_predictor_split()
        models.fit_model(epochs=epochs, batch_size=batch_size)
        predicts = []


        predicts = []
        ##saving the results on the predictions matrix
        interations = 0
        for i in range(input_dim, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < input_dim:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,input_dim)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,input_dim)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break

        product_time_series.predictions[j] = np.array(predicts).flatten()

    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Grasp_RBF - centroids={units}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='value_high',filepath=save_path)


    ##calculating midpoint 
##########################################
for product_name in midpoint_products:
    dataset_name = product_name
    evaluate_df = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    for j in range(product_time_series.n_predictions):
    
    ##calling the model
    
        models = rbf_model(product_time_series.train_windows[j], units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
        models.scale()
        models.predictive_predictor_split()
        models.fit_model(epochs=epochs, batch_size=batch_size)
        predicts = []

        ##saving the results on the predictions matrix
        interations = 0
        for i in range(input_dim, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < input_dim:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,input_dim)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,input_dim)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break
        product_time_series.predictions[j] = np.array(predicts).flatten()


    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Grasp_RBF - centroids={units}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='midpoint',filepath=save_path)


##############################################################
## calculating low point
for product_name in value_low_products:
    dataset_name = product_name
    evaluate_df = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    for j in range(product_time_series.n_predictions):
        ##calling the model
        models = rbf_model(product_time_series.train_windows[j], units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
        models.scale()
        models.predictive_predictor_split()
        models.fit_model(epochs=epochs, batch_size=batch_size)
        predicts = []
        ##saving the results on the predictions matrix
        for i in range(input_dim, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < input_dim:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,input_dim)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,input_dim)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break
        product_time_series.predictions[j] = np.array(predicts).flatten() 
    
    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Grasp_RBF - centroids={units}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='low_point',filepath=save_path)


#%%
'''
k-means rbf
'''



value_description = 'M'
gamma = 0.01
n_input = 5
n_clusters = 10

##calculating high_point 
for product_name in value_high_products:
        
    dataset_name = product_name
    evaluate_df = value_high_df[value_high_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size,window_step=window_step, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    
    for j in range(product_time_series.n_predictions):
        exception = False
        ##calling the model
        
        models = kmeans_rbf(serie=product_time_series.train_windows[j], n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
        models.predictive_predictor_split(n_input)
        models.scale()
        try:
            models.model_fit()
        except:
            print("Singular Matrix found")
            exception = True
            continue
        predicts = []
        ##saving the results on the predictions matrix
        interations = 0
        for i in range(n_input, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break
        if exception == True:
            continue
        product_time_series.predictions[j] = np.array(predicts).flatten()

    delete_indexes = []
    for i in range(len(product_time_series.predictions)):
        if product_time_series.predictions[i,0] == 0:
            delete_indexes.append(i)
    product_time_series.predictions = np.delete(product_time_series.predictions,delete_indexes, axis=0)
    product_time_series.test_windows = np.delete(product_time_series.test_windows,delete_indexes, axis=0)
    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'kmeans_RBF - centroids={n_clusters}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='value_high',filepath=save_path)

##calculating low_point
for product_name in value_low_products:

    dataset_name = product_name
    evaluate_df = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size,window_step=window_step, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()

     

    for j in range(product_time_series.n_predictions):
        exception = False
        ##calling the model
        
        models = kmeans_rbf(serie=product_time_series.train_windows[j], n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
        models.predictive_predictor_split(n_input)
        models.scale()
        try:
            models.model_fit()
        except:
            print("Singular Matrix found")
            exception = True
            continue
        predicts = []
        ##saving the results on the predictions matrix
        interations = 0
        for i in range(n_input, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break
        if exception == True:
            continue
        product_time_series.predictions[j] = np.array(predicts).flatten()

    delete_indexes = []
    for i in range(len(product_time_series.predictions)):
        if product_time_series.predictions[i,0] == 0:
            delete_indexes.append(i)
    product_time_series.predictions = np.delete(product_time_series.predictions,delete_indexes, axis=0)
    product_time_series.test_windows = np.delete(product_time_series.test_windows,delete_indexes, axis=0)
    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'kmeans_RBF - centroids={n_clusters}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='value_low',filepath=save_path)
    
    
    # ## calculating midpoint 

for product_name in midpoint_products:

    dataset_name = product_name
    evaluate_df = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size,window_step=window_step, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    ##calculating high_point 
    for j in range(product_time_series.n_predictions):
        exception = False
        ##calling the model
        
        models = kmeans_rbf(serie=product_time_series.train_windows[j], n_clusters=n_clusters,input_shape=n_input, gamma=gamma)
        models.predictive_predictor_split(n_input)
        models.scale()
        try:
            models.model_fit()
        except:
            print("Singular Matrix found")
            exception = True
            continue
        predicts = []
        ##saving the results on the predictions matrix
        interations = 0
        for i in range(n_input, 0, -1):
            real_values = product_time_series.train_windows[j][-(i):]
            # print(f'Real_Values: {real_values}')
            if i < n_input:
                input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,n_input)
            else: 
                input_predict = product_time_series.train_windows[j][-(i):].reshape(-1,n_input)
            # print(f'input_predict={input_predict}')
            scaled_pred = models.predict(input_predict)
            predict = models.inverse_scale(scaled_pred)
            predicts.append(predict)
            interations += 1
            if interations == 3:
                break
        if exception == True:
            continue
        product_time_series.predictions[j] = np.array(predicts).flatten()

    delete_indexes = []
    for i in range(len(product_time_series.predictions)):
        if product_time_series.predictions[i,0] == 0:
            delete_indexes.append(i)
    product_time_series.predictions = np.delete(product_time_series.predictions,delete_indexes, axis=0)
    product_time_series.test_windows = np.delete(product_time_series.test_windows,delete_indexes, axis=0)
    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'kmeans_RBF - centroids={n_clusters}, gamma={gamma}',PRICE_TYPE_DESCRIPTION='midpoint',filepath=save_path)


# %%

'''
BASELINE
'''


value_description = 'M'
step = 1


##calculating high_point 
for product_name in value_high_products:
        
    dataset_name = product_name
    evaluate_df = value_high_df[value_high_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    
    for j in range(product_time_series.n_predictions):
        baseline = random_walk(step=step, data=product_time_series.train_windows)
        random_walk_fitting_array = baseline.fit()
        predict_array =[product_time_series.train_windows[j][-step:] for i in range(len(product_time_series.test_windows[j])+1)]
        forecast = baseline.predict(predict_array)
        product_time_series.predictions[j] = forecast

    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Baseline',PRICE_TYPE_DESCRIPTION='value high',filepath=save_path)

## low point

for product_name in value_low_products:
        
    dataset_name = product_name
    evaluate_df = value_low_df[value_low_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    
    for j in range(product_time_series.n_predictions):
        baseline = random_walk(step=step, data=product_time_series.train_windows)
        random_walk_fitting_array = baseline.fit()
        predict_array =[product_time_series.train_windows[j][-step:] for i in range(len(product_time_series.test_windows[j])+1)]
        forecast = baseline.predict(predict_array)
        product_time_series.predictions[j] = forecast

    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Baseline',PRICE_TYPE_DESCRIPTION='value high',filepath=save_path)



## midpoint

for product_name in midpoint_products:
        
    dataset_name = product_name
    evaluate_df = midpoint_df[midpoint_df.CODE_DISPLAY_NAME == product_name]
    product_time_series = TimeSeries(name=product_name, series=evaluate_df['VALUE'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
    product_time_series.create_windows()
    
    for j in range(product_time_series.n_predictions):
        baseline = random_walk(step=step, data=product_time_series.train_windows)
        random_walk_fitting_array = baseline.fit()
        predict_array =[product_time_series.train_windows[j][-step:] for i in range(len(product_time_series.test_windows[j])+1)]
        forecast = baseline.predict(predict_array)
        product_time_series.predictions[j] = forecast

    product_time_series.evaluate_model()
    product_time_series.save_data(model=f'Baseline',PRICE_TYPE_DESCRIPTION='value high',filepath=save_path)
            

# %%

product_time_series.n_predictions 
# %%
product_time_series.test_windows
# %%
product_time_series.total_step
product_time_series.step

product_time_series.total_step/product_time_series.step
# %%
product_time_series.total_step % product_time_series.step
