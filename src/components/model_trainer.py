import os
import pandas as pd
import numpy as np
import calendar
import itertools
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

trained_model_file_path = os.path.join('..','..','Results','model.pkl')

def split_train_val_test(X, y):
    """ Split train, validation and test, where
        train = 80%, validation = 10%, test = 10% """
    
    logging.info('Initiating Train, Validation, and Test Split')
    X_train, y_train = X[:2786], y[:2786]
    X_val, y_val = X[2786:3135], y[2786:3135]
    X_test, y_test = X[3135:], y[3135:]
    logging.info('Train, Validation, and Test Split Finished')

    return X_train, y_train, X_val, y_val, X_test, y_test

def data_normalization(X_train, y_train, X_val, y_val, X_test, y_test):
    """ The dataset variables has different scales, and that´s not good
        for the model, so here we normalize all data between 0 and 1, in
        order the variables to have the same weight """

    logging.info('Initiating Data Normalization')
    # Fit the scaler using only the training data
    scaler_X = MinMaxScaler()
    X_train_norm = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Fit and transform
    X_val_norm = scaler_X.transform(X_val.reshape(X_val.shape[0], -1))
    X_test_norm = scaler_X.transform(X_test.reshape(X_test.shape[0], -1))

    # Reshape the data back to original shape
    X_train_norm = X_train_norm.reshape(X_train.shape)
    X_val_norm = X_val_norm.reshape(X_val.shape)
    X_test_norm = X_test_norm.reshape(X_test.shape)

    # For the target variable
    scaler_y = MinMaxScaler()
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    logging.info('Data Normalization Done')

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_y

def LSTM_model(X_train_norm, X_val_norm, y_train_norm, y_val_norm):
    
    logging.info('Initiating LSTM Model Training')
    model = Sequential()
    model.add(InputLayer((14, 11))) # 14 samples, 11 variables
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    print(model.summary())
    print('\n')

    # Compile the model
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[RootMeanSquaredError()]
                  )
    
    model_fit = model.fit(X_train_norm, y_train_norm,
                              validation_data=(X_val_norm, y_val_norm),
                              epochs=10
                              )
    print(model_fit)
    logging.info('LSTM Model Trained')

    return model_fit


# save_object(
#     file_path = self.model_trainer_config.trained_model_file_path,
#     obj = model
# )

def evaluate_model(model, X_test_norm, y_test_norm, scaler_y):
    """The model is tested with X_test_norm dataset. After that the data is denormalized to have the values back again in
       normal kW scale, and then the predictions are evaluated with two evaluation metrics, r2_score and rmse"""

    logging.info('Start Evaluation process')
    model = load_model('model/')
    y_test_predicted_norm = model.predict(X_test_norm).flatten()
    y_test_predicted_original_scale = scaler_y.inverse_transform(y_test_predicted_norm.reshape(-1, 1)).flatten()
    y_test_actual_original_scale = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()
    
    plot = plot_predictions(model, y_test_predicted_original_scale, y_test_actual_original_scale, start=0, end=155)
    
    rmse = np.sqrt(mean_squared_error(y_test_actual_original_scale, y_test_predicted_original_scale))
    prct_rmse = round(rmse*100,2)
    print("- Root Mean Squared Error: {} %".format(prct_rmse))
    r2_square = r2_score(y_test_actual_original_scale, y_test_predicted_original_scale)
    prct_r2_square = round(r2_square*100,2)
    logging.info("- R2 Score: {} %".format(prct_r2_square))
    print(plot)
    logging.info('Evaluation process Finished')
    return prct_rmse, prct_r2_square

def plot_predictions(model, X_test, y_test, start=0, end=155):
    df = pd.DataFrame(data={'Predictions':X_test, 'Actuals':y_test})
    plt.plot(df['Predictions'][start:end], label='PV Forecast')
    plt.plot(df['Actuals'][start:end], label='PV Actual')

    plt.xlabel('Time Periodicity')
    plt.ylabel('kW')
    plt.title('PV Production Forecast Analysis')
    plt.legend(loc='upper right')
    plt.show()
    return 

