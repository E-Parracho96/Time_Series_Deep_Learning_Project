import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def feat_eng1(df):
    
    ''' Here I wil create three new variables. The first two will be a simple shift
        of the day before and two days before at the correspondent timestamp,
        and the third will be a feature that makes the average of production_PV
        at each timestamp over the last 7 days'''
    try:
        logging.info('Applying feat_eng1')
        # Shiftting values
        one_day = 24
        two_days = 48
        df['production_PV_24h'] = df['production_PV'].shift(one_day)
        df['production_PV_48h'] = df['production_PV'].shift(two_days)
        
        # Creating a Feature that makes the average of production_PV at each timestamp over the last 7 days
        df['7days_mean_PV'] = np.nan
        # creates empty matrix for storing values
        temps = pd.DataFrame([])
        N = 7
        for day in range(1,N+1):
            # reorganizes data for vectorial calculation
            temps['day '+str(day)] = df['production_PV'].shift(one_day*day)
        df['7days_mean_PV'] = temps.mean(axis=1)
        df.loc[0:(N*one_day)-1,'7days_mean_PV'] = np.nan

        logging.info('The feat_eng1 is completed')
        return df
    
    except Exception as e:
        raise CustomException(e,sys)

def feat_eng2(df):
    
    ''' Creating two variables, the sum PV of day before
        and sum of standard deviation of the day before '''
    try:
        logging.info('Applying feat_eng2')
        # Set timestamps as index
        df_1 = df.set_index('event_timestamp')
        # Daily sums
        df_2 = df_1.groupby(pd.Grouper(freq = '1D')).sum().shift(1)
        # Daily standard deviations
        df_3 = df_1.groupby(pd.Grouper(freq = '1D')).std().shift(1)

        # New Inputs
        df_1['sum_pv_day_before'] = np.nan
        df_1['std_pv_day_before'] = np.nan

        for doy in range (1,365+1): # doy means day of year

            # -2 refers to 'previous day sum' column
            # -1 refers to 'standard deviation day before' column

            df_1.iloc[(df_1.index.dayofyear == doy),-2] = df_2.iloc[(df_2.index.dayofyear == doy),0][0] 
            df_1.iloc[(df_1.index.dayofyear == doy),-1] = df_3.iloc[(df_3.index.dayofyear == doy),0][0] 
        
        logging.info('The feat_eng2 is completed')
        return df_1
    except Exception as e:
        raise CustomException(e,sys)

def feat_eng3(df):
    
    ''' Converting the time through sine and cosine in order
        to preserve their cyclical significance'''
    try:
        logging.info('Applying feat_eng3')
        # We normalize the hour values to match with the 0-2π cycle
        df["hour_norm"] = 2 * np.pi * df["hour"] / df["hour"].max()
        
        df["cos_hour"] = np.cos(df["hour_norm"])
        df["sin_hour"] = np.sin(df["hour_norm"])

        df = df.drop(['hour', 'hour_norm'], axis=1)
        
        n_cols = df.shape[1]
        df_cols = list(df.columns)
        logging.info('The feat_eng3 is completed')
        logging.info(f'The dataframe that is going to feed the model has {n_cols} columns, and they are: {df_cols}')
        return df
    
    except Exception as e:
        raise CustomException(e,sys)






