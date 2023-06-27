# ExxonMobil-UFPR_AKHub-Time_Series_Forecasting_Project
This is part of a project being developed by UFPR students in partnership with Exxon Mobil. This project mains to forecast time-series values, in special, base-oils.
For this we've made an exploratory analisys of the series and pre-processed the data to fit classical statistics models like ARIMA and Exponential Smoothing.
Then, we've made an sliding windows algorithm to validade the results, saving the scores, and relevant data in an excel file.

---------------------------------------------------------------------------------------------------------------------------------------------------------------
PROJECT STRUCTURE

In this repository you'll find 3 files.

CSV RESULTS:

This file contains the results of sliding windows: there you'll find an excel file containing:
  - RMSE, MAE and MAPE of first, second and third predictions
  - MDA (mean directional acuracy)
  - REGRESSION MDA

OPEN DATASETS:

This file contains the datasets that the model consumed to generate the resuls.

SCRIPTS:

On scripts file we have two another files:

  - Data Pre-processing:
      * This script is the pre-processing algorithm that we utilized on ARGUS dataset, not necessary when working with the crude-oil open dataset.
   
  -  Models Scripts.
  *  There we have:
      - utility's scripts, where the sliding windows and scores are generated;
      - models scripts, where the forecast models are defined;
      - demonstration file, where we generate the results itself.
