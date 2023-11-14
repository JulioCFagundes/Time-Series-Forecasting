#%%

## IMPORTING MODULES
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Layer
from keras import backend as K
from keras.initializers import RandomUniform
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd


#%%



class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(input_shape[1], self.units),
                                  initializer=RandomUniform(minval=0.0, maxval=1.0),
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class rbf_model(RBFLayer):
    def __init__(self, data, units, gamma, input_dim, output_dim):
        self.data = data
        self.units = units
        self.gamma = gamma
        self.input_dim = input_dim
        self.output_dim = output_dim

        Train_lenght = round(len(data))
        self.Train_data = data[:Train_lenght]
        # self.Test_data = data[:Train_lenght]
        self.scaled = False

    def scale(self):
        
        scaler = MinMaxScaler()
        self.train_data_scaled = scaler.fit_transform(self.Train_data.reshape(-1, 1))
        # self.test_data_scaled = scaler.transform(self.Test_data.reshape(-1, 1))
        self.scaled= True
        self.scaler = scaler
    
    def inverse_scale(self,data):
        unscaled_data = self.scaler.inverse_transform(data)
        return unscaled_data

    def predictive_predictor_split(self):
        window_size = self.input_dim
        if self.scaled == True:
            Train_data = self.train_data_scaled[:,0]
            # Test_data = self.test_data_scaled[:,0]
        else:
            Train_data = self.Train_data
            # Test_data = self.Test_data


        X_train = []
        y_train = []
        for i in range(len(Train_data) - window_size):
            X_train.append(Train_data[i:i+window_size])
            y_train.append(Train_data[i+window_size])
        # Imprimir os dados de treinamento
        # for i in range(len(X_train)):
        #     print('Amostra', i+1)
        #     print('Entrada (X_train):', X_train[i])
        #     print('Saída (y_train):', y_train[i])
        #     print('---')

        # X_test = []
        # y_test = []
        # for i in range(len(Test_data) - window_size):
        #     X_test.append(Test_data[i:i+window_size])
        #     y_test.append(Test_data[i+window_size])
        # Imprimir os dados de treinamento
        # for i in range(len(X_test)):
        #     print('Amostra', i+1)
        #     print('Entrada (X_test):', X_test[i])
        #     print('Saída (y_test):', y_test[i])
        #     print('---')


        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        # self.X_test = np.array(X_test)
        # self.y_test = np.array(y_test)

    



    def fit_model(self, epochs, batch_size):

        rbf_model = Sequential()

        # Adição da camada RBF
        rbf_model.add(RBFLayer(units=self.units, gamma=self.gamma, input_shape=(self.input_dim,)))

        # Adição de uma camada densa de saída
        rbf_model.add(Dense(units=self.output_dim, activation='linear'))

        # Compilação do modelo
        rbf_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stopping_callback = EarlyStopping(monitor='loss',min_delta=0.0001, patience=10, restore_best_weights=True)
        rbf_model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping_callback])
        self.model_fit = rbf_model
        return rbf_model

    def predict(self, data):
        predictions = self.model_fit.predict(data)
        self.predictions = predictions
        return predictions



#%%


# input_dim = 5
# output_dim = 1
# # Criação do modelo sequencial
# model = Sequential()

# # Adição da camada RBF
# model.add(RBFLayer(units=50, gamma=0.2, input_shape=(input_dim,)))

# # Adição de uma camada densa de saída
# model.add(Dense(units=output_dim, activation='linear'))

# # Compilação do modelo
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])


# #%%

# # Dados de treinamento e teste
# df = pd.read_csv('./Crude_Oil.csv')
# df.rename(columns={' Value': 'Value'}, inplace=True)
# df.set_index("Date",inplace=True)
# df.index = pd.to_datetime(df.index)
# df.dropna(inplace=True)
# df = df.resample('M').agg('mean')

# # Definir parâmetros para a janela deslizante
# window_size = input_dim  # Tamanho da janela



# train_size = round(0.8*len(df['Value'].values))
# Train_data= df.Value.values[:train_size]
# Test_data = df.Value.values[train_size:]



# scaler = MinMaxScaler()
# train_data_scaled = scaler.fit_transform(Train_data.reshape(-1, 1))
# test_data_scaled = scaler.transform(Test_data.reshape(-1, 1))

# ## Separando variáveis preditoras e preditivas 

# train_data_scaled = train_data_scaled[:,0]
# test_data_scaled = test_data_scaled[:,0]

# X_train = []
# y_train = []
# for i in range(len(train_data_scaled) - window_size):
#     X_train.append(train_data_scaled[i:i+window_size])
#     y_train.append(train_data_scaled[i+window_size])
# # Imprimir os dados de treinamento
# for i in range(len(X_train)):
#     print('Amostra', i+1)
#     print('Entrada (X_train):', X_train[i])
#     print('Saída (y_train):', y_train[i])
#     print('---')

# X_test = []
# y_test = []
# for i in range(len(test_data_scaled) - window_size):
#     X_test.append(test_data_scaled[i:i+window_size])
#     y_test.append(test_data_scaled[i+window_size])
# # Imprimir os dados de treinamento
# for i in range(len(X_test)):
#     print('Amostra', i+1)
#     print('Entrada (X_test):', X_test[i])
#     print('Saída (y_test):', y_test[i])
#     print('---')



# # Escalonamento min-max nos dados de treino e teste


# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

# #%% 
# model.fit(X_train, y_train, epochs=1000, batch_size=10)

# # %%


# # Avaliar o ajuste do modelo
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Previsões do modelo para os dados de treinamento e teste
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)

# # Calcular o MSE e MAE para os dados de treinamento e teste
# mse_train = mean_squared_error(y_train, y_train_pred)
# mae_train = mean_absolute_error(y_train, y_train_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
# mae_test = mean_absolute_error(y_test, y_test_pred)

# # Imprimir as métricas de ajuste
# print('Ajuste do Modelo:')
# print('MSE (Treinamento):', mse_train)
# print('MAE (Treinamento):', mae_train)
# print('MSE (Teste):', mse_test)
# print('MAE (Teste):', mae_test)


# #%% 
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.lineplot(y_train)
# sns.lineplot(y_train_pred)
# plt.show()


# #%%
# y_test_pred_scaled = model.predict(X_test)
# y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
# y_test = scaler.inverse_transform(y_test.reshape(-1,1))
# #%%
# # Plot dos dados reais (y_test) e previstos (y_test_pred)
# sns.lineplot(y=y_test[:,0], x=df.index[train_size + window_size:], color='blue', label='Dados Reais')
# sns.lineplot(y=y_test_pred[:,0], x = df.index[train_size + window_size:], color='red', label='Dados Previstos')

# # Configurações do gráfico

# plt.xlabel('Meses')
# plt.ylabel('Preço')
# plt.title('Dados Reais vs. Dados Previstos')
# plt.legend()

# # Configurar espaçamento dos valores do eixo x
# plt.xticks( rotation=45)
# plt.show()


# #%%
# y_test
