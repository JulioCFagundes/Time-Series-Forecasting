import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from math import exp
from sklearn.preprocessing import MinMaxScaler

class kmeans_rbf():

    def __init__(self, serie, n_clusters, input_shape, gamma):
        self.n_clusters = n_clusters
        self.serie = serie
        self.gamma = gamma
        self.input_shape = input_shape
        self.scaled = False
        self.scaler = MinMaxScaler()

    def predictive_predictor_split(self, n_input):
        X_train = []
        y_train = []

        train_historic_prices = self.serie
        for i in range(len(train_historic_prices) - n_input):
            X_train.append(list(train_historic_prices[i:i+n_input]))
            y_train.append(train_historic_prices[n_input+i])

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        reshaped_X = np.array(X_train).reshape(-1, n_input)
        reshaped_y = np.array(y_train)

        df_columns=[f'lag{i}' for i in range(n_input, 0, -1)]

        new_train_df = pd.DataFrame(reshaped_X,columns = df_columns)
        new_y_df = pd.DataFrame(reshaped_y,columns = ['answer'])
        self.train_df = new_train_df
        self.test_df = new_y_df
        return (new_train_df, new_y_df)
    
    def scale(self):
        
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train.reshape(-1, 1)).reshape(-1,self.input_shape)
        self.y_train_scaled = self.scaler.transform(self.y_train.reshape(-1, 1))
        self.scaled= True
        
    
    def inverse_scale(self,data):
        unscaled_data = self.scaler.inverse_transform(data)
        return unscaled_data

    def Gaussian(self, a, b):
        return exp(-a*b)
    
    def pseudo_inverse(self, G):
        Transposed_G = np.array(G).T
        Transposed_g_dot_G = np.dot(Transposed_G,G)
        self.GG_T = Transposed_g_dot_G
        inverse_GG_T = np.linalg.inv(Transposed_g_dot_G)
        Pseudo_Inverse = np.dot(inverse_GG_T,Transposed_G)
        return Pseudo_Inverse
    
    def model_fit(self):
        gamma = self.gamma
        X_train = self.X_train 
        y_train = self.y_train
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++',  random_state=0, copy_x=True, algorithm='lloyd').fit(self.train_df)
        centroids = kmeans.cluster_centers_
        self.centroids = centroids
        if self.scaled == True:
            X_train = self.X_train_scaled
            y_train = self.y_train_scaled
        
        D_matrix = np.array(distance_matrix(X_train, centroids))
        vecfunc = np.vectorize(self.Gaussian)
        self.vecfunc = vecfunc
        G=vecfunc(D_matrix,gamma)
        G_inverse = self.pseudo_inverse(G)
        Weights = np.dot(G_inverse,y_train)
        self.Weights = Weights

        self.Model_estimates = np.dot(G, Weights)

        return Weights
    

    def predict(self, pred_input, scale: bool = True):
        gamma = self.gamma
        vecfunc = self.vecfunc
        Weights = self.Weights
        if self.scaled == True:
            pred_input = self.scaler.transform(pred_input.reshape(-1, 1)).reshape(-1,self.input_shape)
        D_matrix_predict = distance_matrix(pred_input, self.centroids)
        G_predict=vecfunc(D_matrix_predict,gamma)
        Y_forecast = np.dot(G_predict, Weights)
        return Y_forecast


