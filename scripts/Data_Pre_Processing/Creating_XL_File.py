#%%
import pandas as pd
import os

## Joining the series with historical values of basestocks with Exxon Mapping dataset
argus_dataset_historical_series = pd.read_excel('../../closed_datasets/argus_1.xlsx')

argus_dataset_mapping = pd.read_excel('../../closed_datasets/Argus_BaseStocks.xlsx', sheet_name='Argus to ExxonMobil Mapping')

merged_df = argus_dataset_historical_series.merge(argus_dataset_mapping[['CODE','Region ','Market','Product Name', 'Product Group']], how="inner", on='CODE')
merged_df


#%%

## Like defined on meeting the same itens with multiple unit referencies, itens with quarter tag in the name and itens with negative values will be droped
<<<<<<< HEAD
new_df = merged_df[~merged_df['CODE_DISPLAY_NAME'].isin(['Base oil Group 1 700 US Paulsboro Refining (formerly Valero) east coast $/USG','Base oil Group 1 SN 500 Asia-Pacific quarter', 'Base oil Group 1 SN 500 Asia-Pacific forward premium to gasoil quarter', 'Base oil Group 1 SN 500 cfr Northeast Asia inc. taxes and duty yuan/t', 'Base oil Group 1 SN 500 IOC prices India Chennai domestic Rs/l', 'Base oil Group 1 SN 500 IOC prices India Mumbai domestic Rs/l', 'Base oil Group 1 SN 500 fob NWE quarter','Base oil Group 1 SN 500 fob NWE premium to gasoil quarter', 'Base oil Group 1 SN 500 fob United States $/USG', 'Base oil Group 1 SN 500 US domestic $/USG',  'Base oil Group 1 SN 500 fob United States $/t quarter','Base oil Group 1 SN 500 fob United States $/USG month', 'Base oil Group 1 SN 500 fob United States $/USG quarter', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/t quarter', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/USG month', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/USG quarter', 'Base oil Group 1 SN 400 China NE Fushun domestic yuan/t', 'Base oil Group 1 600 US Calumet Shreveport $/USG', 'Base oil Group I SN 500 Pemex posted Salamanca MXN/t', 'Base oil Group I SN 500 Pemex posted Salamanca $/USG','Base oil Group I SN 650 Pemex posted Salamanca MXN/t','Base oil Group I SN 650 Pemex posted Salamanca $/USG','Base oil Group I SN 500 Turkey domestic ex-tank TL/t', 'Base oil Naftan SN 500 fca EUR/t', 'Base oil Group I SN 300 Brazil domestic reference BRL/l','Base oil Group I SN 500 Brazil domestic reference BRL/l'])].copy()
=======
new_df = merged_df[~merged_df['CODE_DISPLAY_NAME'].isin(['Base oil Group 1 SN 500 Asia-Pacific quarter', 'Base oil Group 1 SN 500 Asia-Pacific forward premium to gasoil quarter', 'Base oil Group 1 SN 500 cfr Northeast Asia inc. taxes and duty yuan/t', 'Base oil Group 1 SN 500 IOC prices India Chennai domestic Rs/l', 'Base oil Group 1 SN 500 IOC prices India Mumbai domestic Rs/l', 'Base oil Group 1 SN 500 fob NWE quarter','Base oil Group 1 SN 500 fob NWE premium to gasoil quarter', 'Base oil Group 1 SN 500 fob United States $/USG', 'Base oil Group 1 SN 500 US domestic $/USG',  'Base oil Group 1 SN 500 fob United States $/t quarter','Base oil Group 1 SN 500 fob United States $/USG month', 'Base oil Group 1 SN 500 fob United States $/USG quarter', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/t quarter', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/USG month', 'Base oil Group 1 SN 500 fob United States premium to heating oil $/USG quarter', 'Base oil Group 1 SN 400 China NE Fushun domestic yuan/t', 'Base oil Group 1 600 US Calumet Shreveport $/USG', 'Base oil Group I SN 500 Pemex posted Salamanca MXN/t', 'Base oil Group I SN 500 Pemex posted Salamanca $/USG','Base oil Group I SN 650 Pemex posted Salamanca MXN/t','Base oil Group I SN 650 Pemex posted Salamanca $/USG','Base oil Group I SN 500 Turkey domestic ex-tank TL/t', 'Base oil Naftan SN 500 fca EUR/t', 'Base oil Group I SN 300 Brazil domestic reference BRL/l','Base oil Group I SN 500 Brazil domestic reference BRL/l'])].copy()
>>>>>>> fee16cd45c8185069acf949c1dc6c7249a1b5d6a
drop_series = new_df[new_df.VALUE <= 0].CODE.unique()
new_df = new_df[~new_df.CODE.isin(drop_series)]
display(new_df)

new_df = new_df[new_df.OPR_DATE.dt.year >= 2021]
display(new_df.PRICE_TYPE_DESCRIPTION)
#%%
new_df.set_index('OPR_DATE', inplace=True)
display(new_df)
#%%
price_description = ['value high', 'value low', 'midpoint']
grouping_frequency = ['M', 'W']
Product_Name =['C600', 'EHC65', 'G34cst']
#%%
filepath = './treated_datasets/new_argus_series.xlsx'
#%%
for product in Product_Name:
    product_series = new_df[new_df['Product Name'] == product].copy()
    print('essa é a product_series: ',product_series)

    listed_base_oils = product_series.CODE_DISPLAY_NAME.unique()

    for base_oil in listed_base_oils:
        base_oil_filtered_df = product_series[product_series.CODE_DISPLAY_NAME == base_oil]

        for frequency in grouping_frequency:
            for description in price_description:
                ## Setting index date
                df_to_excel = base_oil_filtered_df.reset_index()
                df_to_excel.set_index('OPR_DATE', inplace=True)
                ## Setting midpont, high or low values
                df_to_excel = df_to_excel[df_to_excel['PRICE_TYPE_DESCRIPTION'] == description]
                # df_to_excel.drop_duplicates(inplace=True)
                ## Grouping data by week, month etc. (grouping patterns defined on "Frequency")
                df_to_excel = df_to_excel.resample(frequency).agg('mean')
                ## calling back the variables lost on the resample
                CODE_NAME = [base_oil for i in range(len(df_to_excel))]
                Value_Type = [description for i in range(len(df_to_excel))]
                Frequency = [frequency for i in range(len(df_to_excel))]
                Exxon_Product = [product for i in range(len(df_to_excel))]
                df_to_excel['CODE_NAME'] = CODE_NAME
                df_to_excel['value_description'] = Value_Type
                df_to_excel['Frequency'] = Frequency
                df_to_excel['Exxon_Product'] = Exxon_Product
                df_to_excel.reset_index(inplace=True)
                # Appends data to output file if it exists. Otherwise, creates it.
                file_exists = os.path.exists(filepath)
                # Appends data to output file if it exists. Otherwise, creates it.
                if file_exists:
                    df_to_excel.to_csv(filepath, index=False, mode='a', header=False)
                else:
                    df_to_excel.to_csv(filepath, index=False)

# %%
df_to_excel.display()
