#%%
import pandas as pd
# The data is spread across three files, so we'll concatenate them into one DataFrame
data_path = r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast - Copia\closed_datasets\argus'
df = pd.concat([pd.read_excel(data_path + f'_{i}.xlsx') for i in range(1, 4)])
df.head()
# %%
# Splits unit of measure from currency
df['CURRENCY'] = df['UNIT'].str.split('/').str[0]
df['UOM'] = df['UNIT'].str.split('/').str[1] # UoM stands for Unit of Measure
df = df.drop(columns=['UNIT'])
df.head()
# %%
# Keeps only series in USD
# No data loss here as the series in other currencies are also present in USD in this dataset
currency_filter = df['CURRENCY'] == 'USD'
# Keeps only series in Tons. We do not have the density information of the products 
# (not all of them are ExxonMobil made)
uom_filter = df['UOM'] == 't'
# Removed series that have a negative values
min_val_by_code = df.groupby(['CODE'])['VALUE'].min()
codes_with_negative_vals = min_val_by_code[min_val_by_code < 0].index
negative_filter = ~df['CODE'].isin(codes_with_negative_vals)
# Applies all filters
df = df[currency_filter & uom_filter & negative_filter]
# %%
# Creates a month-end field based on the OPR_DATE. We will use this to aggregate the values to the month level
df['OPR_DATE'] = pd.to_datetime(df['OPR_DATE'], format='%Y-%m-%d')
df['MONTH_END'] = df['OPR_DATE'] + pd.offsets.MonthEnd(0)
df.head()
# %%
# No need to keep UoM and Currency columns as we are keeping only Tosn and USD
columns_to_agg = ['MONTH_END', 'CODE', 'CODE_DISPLAY_NAME', 'PRICE_TYPE_DESCRIPTION']

df_high = df[df['PRICE_TYPE_DESCRIPTION']=='value high'].groupby(columns_to_agg)[['VALUE']].max().reset_index()
df_midpoint = df[df['PRICE_TYPE_DESCRIPTION']=='midpoint'].groupby(columns_to_agg)[['VALUE']].mean().reset_index() 
df_low = df[df['PRICE_TYPE_DESCRIPTION']=='value low'].groupby(columns_to_agg)[['VALUE']].min().reset_index()
df = pd.concat([df_high, df_midpoint, df_low])

del [df_high, df_midpoint, df_low]

df.head()
# %%
data_map_path = r'C:\Users\Work\Desktop\Julio\Exxon\Códigos ARGUS\Exxon_UFPR_project-Time_Series_Forecast - Copia\closed_datasets\Argus_to_ExxonMobil_Mapping.xlsx'
sheet_name = 'Planilha1'
usecols = ['CODE', 'Region ', 'Market', 'Product Name', 'Product Group']
df_map = pd.read_excel(data_map_path, sheet_name=sheet_name, usecols=usecols)
df_map.columns = df_map.columns.str.strip(' ') # Removing leading and trailing blanks from columns names
df_map.head()
# %%

df = df.merge(df_map, on='CODE', how='inner')
df = df.drop_duplicates()
df.head()
# %%
df.to_csv('argus_clean.csv', index=False)
# %%
df
