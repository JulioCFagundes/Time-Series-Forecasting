#%%
import pandas as pd
import numpy as np
import seaborn as sns
import plotly as pt
import matplotlib.pyplot as plt

#%%
results_path = r'C:\Users\User\Desktop\Exxon\git akhub\akhub\JULIO\scripts\Data_Pre_Processing\Cleaned_Datasets\TESTS_RESULTS.csv'
results = pd.read_csv(results_path)
results.model.unique()
new_df = results[['model', 'name', 'mape']]
filter_df = new_df.groupby('model').mean()
df_sorted = filter_df.sort_values(by='mape', ascending=True)
df_sorted.reset_index(inplace=True)
df_sorted_best_ten = df_sorted.iloc[:10]
best_models = df_sorted.iloc[:10].model

results_plot = results[results['model'].isin(best_models)]
#%%
sns.boxplot(data=results_plot, x="mape", y="model")
plt.title('Best 10 Models boxplot')
plt.show()
# %%
#%%
##################### BARPLOT #########################
##### Os 10 melhores




sns.barplot(x=df_sorted_best_ten.model, y=df_sorted_best_ten.mape)
plt.xticks(rotation=45)
plt.title('Best 10 mean mape')
plt.show()

#%%

# %%
len(results.name.unique()[0])