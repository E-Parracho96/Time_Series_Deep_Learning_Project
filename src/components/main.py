from src.components.data_ingestion import data_ingestion
from src.components.data_preprocess import *
from src.components.feature_engineering import *
from src.components.model_logic import forecasting_logic
from src.components.model_trainer import *
from src.exception import CustomException
from src.logger import logging

import sys
import glob as glob
import argparse
import pandas as pd
from tensorflow.keras.models import load_model


def training(filelist):
    try:
        evaluation_metrics = {'RMSE': [], 'R2_SCORE': []}
        df_evaluation_metrics = pd.DataFrame(evaluation_metrics)

        logging.info('Training Function - Initiate Forecast')

        for count, filename in enumerate(filelist):
            print(filename)
            df = data_ingestion(filename, file2_path)

            filename = filename.split('\\')[-1].split('.csv')[0]
            # Validate files with zero PV production
            if df.production_PV.mean() == 0:
                continue

            interpolation = interpolation_nan(df)
            data_variables = time_variables(interpolation)
            data_resample = resample(data_variables)
            ft_eng1 = feat_eng1(data_resample)
            ft_eng2 = feat_eng2(ft_eng1)
            ft_eng3 = feat_eng3(ft_eng2)
            X, y, n_features, days_lookback = forecasting_logic(ft_eng3, frequency=24, days_lookback=14)  # not needed for prediction
            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, train_percent=0.8, val_percent=0.1, test_percent=0.1)
            X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_y = data_normalization_train(
                X_train, y_train, X_val, y_val, X_test, y_test)

            model, checkpoint_path = LSTM_model(X_train_norm, X_val_norm, y_train_norm, y_val_norm, days_lookback, n_features)
            # Loads the weights
            model.load_weights(checkpoint_path)
            # Save the models
            model.save(f'models\{filename}_model.keras')
            prct_rmse, prct_r2_square = evaluate_model(model, X_test_norm, y_test_norm, scaler_y)
            # Create a new DataFrame with RMSE and R2_SCORE values
            result_df = pd.DataFrame({'RMSE': [prct_rmse], 'R2_SCORE': [prct_r2_square]})

            # Append the new DataFrame to df_evaluation_metrics
            df_evaluation_metrics._append(result_df, ignore_index=True)
            df_evaluation_metrics = df_evaluation_metrics.reset_index()
            df_evaluation_metrics = df_evaluation_metrics.drop(['index'], axis=1)

        logging.info('Training Function Finished')

        return df_evaluation_metrics
    
    except Exception as e:
        raise CustomException(e,sys)


def predict(filelist):
    try:
        for count, filename in enumerate(filelist):
            print(filename)

            df = data_ingestion(filename, file2_path)

            df = data_ingestion(filename, file2_path)

            # Validate files with zero PV production
            if df.production_PV.mean() == 0:
                sys.exit()

            interpolation = interpolation_nan(df)
            data_variables = time_variables(interpolation)
            data_resample = resample(data_variables)
            ft_eng1 = feat_eng1(data_resample)
            ft_eng2 = feat_eng2(ft_eng1)
            ft_eng3 = feat_eng3(ft_eng2)
            X, y, n_features, days_lookback = forecasting_logic(ft_eng3, frequency=24, days_lookback=21)  # not needed for prediction
            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, train_percent=0.8, val_percent=0.1, test_percent=0.1)
            X_test_norm = data_normalization_pred(X_test)

            model = load_model(f'models\{filename}_model.keras')
            yhat = model.predict(X_test_norm, verbose=0)
            print(yhat)
            plt.plot(yhat)
            plt.show()
            
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time Series Forecast Model Template')
    parser.add_argument('--files_dir', '-f', help='Data path', required=True)
    parser.add_argument('--file2_path', '-fp', help='Data path', required=True)

    args = parser.parse_args()

    filelist = glob.glob(f'{args.files_dir}/*.csv')
    file2_path = args.file2_path

    training(filelist)
    #predict()
