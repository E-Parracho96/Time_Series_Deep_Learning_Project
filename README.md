## Deep Learning Time Series Project

The goal of this project is to develop an algortihm which is able to make a good forecast of PV production for the day after.
Beacuse this is a Time Series dataset I decided to apply the LSTM (RNN Deep Learning) model, beacuse it is designed to capture long-term dependencies and patterns in sequential data.
Here we only have data about PV production over 1 year. The dataset has a record each 15 min. Some other variables were developed to feed the model.

I decided to use the following approach:
To do the forecast, the model will be based in a rolling window, in this case 14 days (it´s flexible to change). So basically, if we want to forecast what will be the PV production tomorrow at 2 PM, the model will look back to the previous 14 records at 2 PM for every feature, and that data will be the input of the LSTM model that will do the forecast.
