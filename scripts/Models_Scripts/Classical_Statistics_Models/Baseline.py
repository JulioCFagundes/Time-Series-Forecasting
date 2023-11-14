#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class random_walk:
    def __init__(self, data, step=1):

        self.step = step
        self.series = data
        

    def fit(self):
        random_walk_fitting_array = [self.series[i - self.step] for i in range(self.step,len(self.series))]
        self.random_walk_fitting_array = random_walk_fitting_array
        
        return random_walk_fitting_array
    
    def predict(self):

        forecast =  [self.series[-self.step] for i in range(3)]
        forecast = np.array(forecast).reshape(len(forecast))


        return forecast

# %%


### Usecase

# dataset_path =  r'C:\Users\User\Desktop\Exxon\Forecasting ARGUS base\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\scripts\Data_Pre_Processing\Cleaned_Datasets\argus_clean.csv'
# save_path= r'C:\Users\User\Desktop\Exxon\Forecasting ARGUS base\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project-main\closed_datasets\Closed_results\TESTS_RESULTS.csv'
# freq = 'Monthly'

# ## Calling the dataset
# dataset = pd.read_csv(dataset_path)

# dataset = dataset[(dataset.CODE == 'PA0007453') & (dataset.PRICE_TYPE_DESCRIPTION == 'value high')] 

# series = dataset['VALUE']

# baseline = random_walk(step=3, data=series)

# random_walk_fitting_array = baseline.fit()

# sns.lineplot(series, label = 'real prices')
# sns.lineplot(random_walk_fitting_array, label = 'forecast')
# plt.show()
# y = pd.Series([1,2,3,4,5,6])

# forecast = baseline.predict(y)

# sns.lineplot(y)
# sns.lineplot(forecast)
# plt.show()
# # %%
# series = dataset['VALUE']
# step=1
# Train_Series = series[:140]
# Test_Series = series[(140-(step+1)):]
# Test_Series = Test_Series.reset_index()
# Test_Series = Test_Series.drop(columns=['index'])




# # %%

# baseline = random_walk(step=step, data=Train_Series)
# random_walk_fitting_array= baseline.fit()
# sns.lineplot(Train_Series[:-step], label = 'real prices')
# sns.lineplot(random_walk_fitting_array, label = 'forecast')
# plt.show()

# forecast = baseline.predict(Test_Series.values)

# sns.lineplot(Test_Series.values[step:], label = 'real prices')
# sns.lineplot(forecast, label = 'forecast')
# plt.show()


# # %%

# Test_Series[:-step]
# # %%
