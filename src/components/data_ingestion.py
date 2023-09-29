import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging


def organize_file (df):
    df = df.rename(columns = {'producao_total':'production_PV','consumo_rede':'consumption','injecao_rede':'grid_injection'}) # rename columns
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], unit='s') # Converting to datetime
    df = df.sort_values(by=['event_timestamp']).reset_index(drop=True) # Sorting the data by datetime

    return df

def data_ingestion(file1_path, file2_path):

    """This function read the house file, the sun_position_PT file and join them."""
    try:
        logging.info('Read the house dataset as dataframe')
        df1 = pd.read_csv(file1_path)

        # Check if essencial columns are in the dataframe  
        column_names = ['event_timestamp', 'producao_total']
        missing_columns = [col for col in column_names if col not in df1.columns]
        if missing_columns:
            raise ValueError(f"The following essential columns are missing in the DataFrame: {', '.join(missing_columns)}")
        
        # Check if producao_total value is float type an if event_timestamp is int
        if  df1['producao_total'].dtype != float and df1['event_timestamp'].dtype != int:
            raise ValueError("Error in column type.")

        df_1 = organize_file(df1)

        # The sun_position_PT CSV has three columns: time, elevation (elev), and azimute (azim)
        df2 = pd.read_csv(file2_path) 
        logging.info('Read the sun_Position_PT dataset as dataframe')

        df2.iloc[0:35041,1:] # The sun_position_PT file has 1 more day, so we have to filter it and also filter the time column
        df_merged = df_1.join(df2)

        logging.info('Ingestion of the data is completed')
        return df_merged
    
    except Exception as e:
        raise CustomException(e,sys)
    
# if __name__ == "__main__":
#     file1_path = os.path.join('Dados\zKFdRou77JuivhIm.csv')
#     file2_path = os.path.join('Dados\sun_position_PT.csv')
    
#     df_result = data_ingestion(file1_path, file2_path)