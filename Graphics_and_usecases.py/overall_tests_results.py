#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%

df = pd.read_csv(r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast\closed_datasets\Closed_results\TESTS_RESULTS.csv')
df = df[df.name == 'Base oil Group 1 SN 500 Asia-Pacific month']
df.set_index('model', inplace=True)
sorted_df = df.sort_values(by='rmse_1st', ascending=True)

df.head()

#%%
######### all models

plot_df = sorted_df[sorted_df['rmse_1st'] <=400]
sns.barplot(y=plot_df['rmse_1st'], x=plot_df.index)
plt.xticks(rotation=70)

#%%

######### top 10
plot_df = df[df.index.isin([
'Arima(2,3,4)',
'Arima(2,1,4)',
'Arima(0,4,4)',
'Arima(2,3,0)',
'Arima(1,3,0)',
'Arima(1,3,4)',
'Arima(0,0,2)',
'Arima(0,5,3)',
'Arima(0,4,1)',
'Arima(0,5,4)',
]

)]

sns.barplot(y=plot_df['rmse_1st'], x=plot_df.index)
plt.xticks(rotation=70)
#%%
sorted_df.iloc[:14]

#%%
best_ten_df = sorted_df[sorted_df.index.isin([
'Arima(2,0,0)',
'Arima(2,1,1)',
'Arima(1,1,0)',
'Arima(2,1,0)',
'Arima(1,1,1)',
'Arima(2,0,1)',
'Arima(1,0,1)',
'Arima(1,0,2)',
'Arima(0,1,1)',
'Arima(2,2,3)'


])]

sns.barplot(y=best_ten_df['rmse_1st'], x=best_ten_df.index)
plt.xticks(rotation=70)



#%% for crude_oil

df = pd.read_csv(r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast\Open_datasets\Crude_Oil.csv')


df.head()