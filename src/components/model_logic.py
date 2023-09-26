import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging


def filter_night_out(df, frequency=24, days_lookback=14): # admitting we have 24 records per hour
    
    """This is a dynamic function. The logic here is having a rolling window of 14 days for each record
       which is already resampled hourly.
       First it will loop for each production_PV record, and the ideia is trying to predict what will be
       the production_PV value in the next day based on the last 14 records at each timestamp.
       
       Example: If I want to predict what will be the production_PV value at 1PM in 15 May, this prediction
                will be based on the last 14 records (between 1 and 14 May) at 1PM, then want to predict
                what will be the value at 2PM, this prediction will be based on the values at 2PM between
                1 and 14 May, ... , skipping to the next day, and so on.
       
       Here the night periods are filtered because otherwise the model would have a big bias, predicting that 
       at night the production would be 0, which is obvious, changing the real performance of the model.
       If we are predicting the production_PV values we only want the diurn part of the dataset, that´s why,
       the sun elevation is filtered at 5 degrees and the production_PV values at 0.0001.
    """
    
    try:
        logging.info('Initializing filter_night_out logic implementaion')
        df_as_np = df.to_numpy()
        X, y = [], []
        total_days = len(df_as_np) // frequency # Number of days to analyse

        for day in range(total_days - days_lookback):
            for hour in range(frequency):  # For each hour of the day
                start_idx = day * frequency + hour
                end_idx = start_idx + days_lookback * frequency
                
                X_slice = df_as_np[start_idx:end_idx:24]  # Select PV value each day at each timestamp
                
                # Skip the iteration if there is any nan value in X_slice
                if np.isnan(X_slice).any():
                    continue
                # Skip the iteration if the sun elevation is <5 and the first value of X_slice is <= 0.0001
                if (X_slice[0][1] < 5) & (X_slice[0][0] <= 0.0001):
                    continue
                
                y_value = df_as_np[end_idx][0]  # Select PV value of the day after at the same timestamp
                
                non_zero_count = np.count_nonzero(X_slice)  # Count the number of elements non-zero
                
                # Add X_slice if the values are different than 0
                if  not np.all(X_slice == 0):
                    X.append(X_slice)
                
                # Condition to decide if we include y_value
                if non_zero_count >= 1:
                    y.append(y_value)

        logging.info('filter_night_out logic completed')
        return np.array(X), np.array(y)
            
    except Exception as e:
        raise CustomException(e,sys)
   