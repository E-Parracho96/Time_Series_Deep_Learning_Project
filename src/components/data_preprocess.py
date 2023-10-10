import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def interpolation_nan(df):
    """Interpolate missing values that occur isolated in the dataset"""
    try:
        logging.info('Beggining nan interpolation')
        df['nan'] = df['production_PV'].isna()
        df['prev_nan'] = df['nan'].shift(fill_value=False)
        df['next_nan'] = df['nan'].shift(-1, fill_value=False)

        # Create a mask for isolated nans
        mask = df['nan'] & ~df['prev_nan'] & ~df['next_nan']

        # Interpolating only where the mask is True
        df.loc[mask, 'production_PV'] = df['production_PV'].interpolate(method='linear', limit=1)

        # Removing columns
        df = df.drop(columns=['nan', 'prev_nan', 'next_nan', 'consumption', 'grid_injection'])
        logging.info('Nan interpolation is completed')

        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def time_variables(df):
    """Creating new columns by splitting the datetime column and exclude June month"""
    
    try:
        logging.info('Beggining Time Variables')
        df['month'] = df['event_timestamp'].dt.month # Creating a month variable
        df['hour'] = df['event_timestamp'].dt.hour # Creating a hour variable

        # Exclude June due to absence of data as observed in EDA
        df = df[df['month']!=6]
        logging.info('Time Variables is completed')

        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def resample(df):
    """Resampling data hourly. For production_PV we get the sum of the 4 records in that hour,
       and for the other variables we get the mean value"""
    
    try:
        logging.info('Initiating Resample process')
        df_resample_sum = df[['event_timestamp', 'production_PV']].resample('60min', on='event_timestamp').sum()
        df_resample_mean = df[['event_timestamp', 'elev', 'azim', 'month', 'hour']].resample('60min', on='event_timestamp').mean()
        df_resample = pd.concat([df_resample_sum, df_resample_mean], axis=1)
        df_resample = df_resample.reset_index()

        df_resample_cols = list(df_resample.columns)
        logging.info('Resample process completed')
        logging.info(f'The dataframe that is going to feed the Feature Engineering script has the following columns: {df_resample_cols}')

        return df_resample
    
    except Exception as e:
        raise CustomException(e, sys)    


    


