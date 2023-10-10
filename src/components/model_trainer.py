import sys
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

from src.exception import CustomException
from src.logger import logging


def split_train_val_test(X, y, train_percent=0.8, val_percent=0.1, test_percent=0.1):
    """ Split train, validation and test, where
        train = 80%, validation = 10%, test = 10% """
    try:
        logging.info('Initiating Train, Validation, and Test Split')
        total_samples = X.shape[0]
        train_samples = int(total_samples * train_percent)
        val_samples = int(total_samples * val_percent)
        test_samples = int(total_samples * test_percent)

        if train_percent + val_percent + test_percent != 1:
                raise ValueError("The sum of train, val, and test should be equal to total_samples.")

        X_train, y_train = X[:train_samples], y[:train_samples]
        X_val, y_val = X[train_samples:train_samples + val_samples], y[train_samples:train_samples + val_samples]
        X_test, y_test = X[train_samples + val_samples:], y[train_samples + val_samples:]
        logging.info('Train, Validation, and Test Split Finished')

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        raise CustomException(e, sys)
    

def data_normalization_train(X_train, y_train, X_val, y_val, X_test, y_test):
    """ The dataset variables has different scales, and that´s not good
        for the model, so here we normalize all data between 0 and 1, in
        order the variables to have the same weight """
    
    try:
        logging.info('Initiating Data Normalization Train')
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
        logging.info('Data Normalization Train Done')

        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_y
    
    except Exception as e:
        raise CustomException(e, sys)


def data_normalization_pred(x):
    try:
        logging.info('Initiating Data Normalization Pred')
        # Fit the scaler using only the training data
        scaler_X = MinMaxScaler()

        x_norm = scaler_X.fit_transform(x.reshape(x.shape[0], -1))  # Fit and transform
        x = x_norm.reshape(x.shape)
        logging.info('Data Normalization Pred Done')

        return x

    except Exception as e:
        raise CustomException(e, sys)


def LSTM_model(X_train_norm, X_val_norm, y_train_norm, y_val_norm, days_lookback, n_features):
    """LSTM means Long Short-Term Memory is a type of recurrent neural network (RNN) architecture
       designed to capture long-term dependencies and patterns in sequential data.
       Here is developed the Model Architecture."""
    
    try:
        logging.info('Initiating LSTM Model Training')
        model = Sequential()
        model.add(InputLayer((days_lookback, n_features)))
        model.add(LSTM(64))
        model.add(Dense(8, 'relu'))
        model.add(Dense(1, 'linear'))

        checkpoint_path = "training_1/cp.ckpt"

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        verbose=1)

        print(model.summary())
        print('\n')

        # Compile the model
        model.compile(loss=MeanSquaredError(),
                    optimizer=Adam(learning_rate=0.001),
                    metrics=[RootMeanSquaredError()]
                    )

        model_fit = model.fit(X_train_norm, y_train_norm,
                            validation_data=(X_val_norm, y_val_norm),
                            epochs=10,
                            callbacks=[cp_callback]
                            )
        print(model_fit)
        logging.info('LSTM Model Trained')

        return model, checkpoint_path
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(model, X_test_norm, y_test_norm, scaler_y):
    """The model is tested with X_test_norm dataset. After that the data is denormalized to have the values back again in
       normal kW scale, and then the predictions are evaluated with two evaluation metrics, r2_score and rmse"""

    try:
        logging.info('Start Evaluation process')
        y_test_predicted_norm = model.predict(X_test_norm).flatten()
        y_test_predicted_original_scale = scaler_y.inverse_transform(y_test_predicted_norm.reshape(-1, 1)).flatten()
        y_test_actual_original_scale = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1)).flatten()

        plot_predictions(y_test_predicted_original_scale, y_test_actual_original_scale)

        rmse = np.sqrt(mean_squared_error(y_test_actual_original_scale, y_test_predicted_original_scale))
        prct_rmse = round(rmse * 100, 2)
        print("- Root Mean Squared Error: {} %".format(prct_rmse))
        r2_square = r2_score(y_test_actual_original_scale, y_test_predicted_original_scale)
        prct_r2_square = round(r2_square * 100, 2)
        print("- R2 Score: {} %".format(prct_r2_square))
        logging.info('Evaluation process Finished')
        
        return prct_rmse, prct_r2_square
    
    except Exception as e:
        raise CustomException(e, sys)


def plot_predictions(X_test, y_test, start=0, end=155):
    df = pd.DataFrame(data={'Predictions': X_test, 'Actuals': y_test})
    plt.plot(df['Predictions'][start:end], label='PV Forecast')
    plt.plot(df['Actuals'][start:end], label='PV Actual')

    plt.xlabel('Time Periodicity')
    plt.ylabel('kW')
    plt.title('PV Production Forecast Analysis')
    plt.legend(loc='upper right')
    plt.show()
    return
