import os
import sys
import glob as glob
from src.components.data_ingestion import data_ingestion
from src.components.data_preprocess import *
from src.components.feature_engineering import *
from src.components.model_logic import forecasting_logic
from src.components.model_trainer import *
from src.exception import CustomException
from src.logger import logging


# Main program
files_dir=r"C:\Users\emparrac\Desktop\Emanuel\Courses Data Science\My_Project_EDP_Data\Dados_EDP\anonimized_3"
file2_path =r'C:\Users\emparrac\Desktop\Emanuel\Courses Data Science\My_Project_EDP_Data\Dados_EDP\sun_position_PT.csv'

#Verificar ficheiros csv na pasta
os.chdir(files_dir) 
filelist = glob.glob("*.csv")

def main_function(filelist):

    evaluation_metrics = {'RMSE':[], 'R2_SCORE':[]}
    df_evaluation_metrics = pd.DataFrame(evaluation_metrics)
    
    for count,filename in enumerate (filelist):    
        #print(filename)
        df = data_ingestion(filename,file2_path)

        # validar files com produções nulas - file 167 com produção anual = 0
        if df.production_PV.mean()==0:
            continue
            
        interpolation = interpolation_nan(df)
        data_variables = time_variables(interpolation)
        data_resample = resample(data_variables)
        ft_eng1 = feat_eng1(data_resample)
        ft_eng2 = feat_eng2(ft_eng1)
        ft_eng3 = feat_eng3(ft_eng2)
        X, y = forecasting_logic(ft_eng3, frequency=24, days_lookback=14)
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_y = data_normalization(X_train, y_train, X_val, y_val, X_test, y_test)
        model = LSTM_model(X_train_norm, X_val_norm, y_train_norm, y_val_norm)
        prct_rmse, prct_r2_square = evaluate_model(model, X_test_norm, y_test_norm, scaler_y)
        
        # Create a new DataFrame with RMSE and R2_SCORE values
        result_df = pd.DataFrame({'RMSE': [prct_rmse], 'R2_SCORE': [prct_r2_square]})
        
        # Append the new DataFrame to df_evaluation_metrics
        df_evaluation_metrics = df_evaluation_metrics.append(result_df, ignore_index=True)
        df_evaluation_metrics = df_evaluation_metrics.reset_index()
        df_evaluation_metrics = df_evaluation_metrics.drop(['index'],axis=1)
        
    return df_evaluation_metrics