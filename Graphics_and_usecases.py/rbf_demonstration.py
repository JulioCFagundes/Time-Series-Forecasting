
#%%
import numpy as np
import pandas as pd
from scripts.Models_Scripts.utility import TimeSeries

# from Classical_Statistics_Models.classical_statistics_models import UFPR_Models
# from Machine_Learning_Models.RBF_model import rbf_model

#%%

dataset_path =  r'C:\Users\User\Desktop\Exxon\Forecasting ARGUS base\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\Open_datasets\Crude_Oil.csv'

freq = 'Monthly'
save_path = '../../closed_datasets/Closed_results/test_results.csv'
dataset_name = 'Crude_Oil'
train_window_size=48
test_window_size=3
fillnamethod = 'forwardfill'
value_description = 'value high'

#%% FOR CRUDE OIL
df = pd.read_csv(dataset_path)
df.rename(columns={' Value': 'Value'}, inplace=True)
df.set_index("Date",inplace=True)
df.index = pd.to_datetime(df.index)
df.dropna(inplace=True)
df = df.resample('M').agg('mean')
#%% FOR ARGUS BASE



#%% 
time_series = TimeSeries(name=dataset_name, series=df['value'], train_window_size=train_window_size, test_window_size=test_window_size, frequency=freq, fillna=fillnamethod, value_description = value_description)
time_series.create_windows()

input_dim = 3
output_dim = 1
units= 30
gamma = 0.1
# train_size = 1
batch_size=50
epochs = 3000

#%% COMPLETE TEST
for j in range(time_series.n_predictions):
    ##calling the model
    models = rbf_model(time_series.train_windows[j], units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
    models.scale()
    models.predictive_predictor_split()
    models.fit_model(epochs=epochs, batch_size=batch_size)
    predicts = []
    ##saving the results on the predictions matrix
    for i in range(3, 0, -1):
        real_values = time_series.train_windows[j][-(i+1):]
        print(f'Real_Values: {real_values}')
        if i < 3:
            input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,input_dim)
        else: 
            input_predict = time_series.train_windows[j][-(i+1):].reshape(-1,input_dim)
        print(f'input_predict={input_predict}')
        scaled_pred = models.predict(input_predict)
        predict = models.inverse_scale(scaled_pred)
        predicts.append(predict)

    time_series.predictions[j] = np.array(predicts).flatten()
    

## calculating scores
time_series.evaluate_model()
## saving results
time_series.save_data(model=f'Simple Exponential Smoothing',filepath=save_path)
   

#%% LAST SLIDING WINDOWS EVALUATE

models = rbf_model(time_series.train_windows[-1], units=units, gamma=gamma, input_dim=input_dim, output_dim=output_dim)
models.scale()
models.predictive_predictor_split()
models.fit_model(epochs=epochs, batch_size=batch_size)
predicts = []
##saving the results on the predictions matrix
for i in range(3, 0, -1):
    real_values = time_series.train_windows[-1][-(i+1):]
    print(f'Real_Values: {real_values}')
    if i < 3:
        input_predict = np.append(np.array(real_values).flatten(),np.array(predicts).flatten()).reshape(-1,input_dim)
    else: 
        input_predict = time_series.train_windows[-1][-(i+1):].reshape(-1,input_dim)
    print(f'input_predict={input_predict}')
    scaled_pred = models.predict(input_predict)
    predict = models.inverse_scale(scaled_pred)
    predicts.append(predict)
time_series.predictions[-1] = np.array(predicts).flatten()
    
#%%
models.fit_model(epochs=1, batch_size=32)

models.predict(time_series.train_windows[j][-input_dim:].reshape(-1,1))

#%%
save_path = '../../closed_datasets/Closed_results/test_results.csv'
time_series.save_data(model=f'RBF',filepath=save_path)