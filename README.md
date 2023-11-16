INTRODUCTION:

This project aims to predict future prices of crude oil derivatives.
Below you can see the structure of the project, respecting good programing practices like files responsabilities, intuitive functions and variables, comments to help users to 
reproduce the project and an intuitive structure. 

On our dataset we have about 300 series that we want to forecast so we had to make a robust algorithm but generic enough to fit a large number of series and retrive us a optimic 
performance. 

In this projects we have two outputs:

 1. Evaluation of the best model in the slidding windows algorithm (it measures the constancy of the model so we have an realiable measure of the model's performance)
 2. Forecast with the choosen model for the next 3 months (it's 3 months because we decided it for this specific problem, but changing params we can forecast as much months (or
any other granularity) as you want. 

PROJECT STRUCTURE

As the datasets are closed, we cannot share it on this repository.

Inside our scripts folder you'll find three folders:

Data_Pre_Processing:

 - This file contains the algorithm developed for our data pre-process. To generate our response, you'll need to input the base-oils historical prices and the algorithm will generate the pre-processed data into Cleaned_Datasets folder. 



Models_Scripts:

On Models_Scripts folder we have five files and two folders. Starting with the files:

     files:
         - demonstration:
             * in this file, we compile all codes in order to make our summarized table with the results of the extensive tests with all base-oils. This file is responsible for creating the CSV file with the models metrics. 

         - utility: 
             * this file stores the classes responsible for generating the sliding windows that are used on our extensive tests, calculation of model's avaliation metrics and for saving the results.

         - UseCases:
             * here, we made a single training and evaluation of the models. In this usecase, we forecast the last 3 months of the database and use the 48 months before to train the models. This process runs for every base-oil on the dataset. 

         - Utility_copy:
             * it's just the Utility file but with some new features in order to run the neural networks on our extensive tests, since it was longing more than weeks to runs on the original utility algorithm.
         
     folders: 
         -  Classical_Statistics_Models:

             This folder is responsible for the classic statistical models codes. Here we can find the code used to adjust arima, exponencial smoothing algorithms and so on... 

             Also here we have our baseline, which is the simplest model that we want all the others to beat its performance (the ones who performs better we can considerate as a good model and the ones who don't we can just throw it out).

         - Machine_Learning_Models:

             This folder is responsible for the machine learning models codes. Here we can find the codes utilized to adjust the Radial Basis Function Neural Network using two approches to find its params: the first is using Grasp method and the second one is utilizing K-means.  

    


Results_Interpretation:
     
     - This file contains the code utilized to plot the barplot of the results for one product.
