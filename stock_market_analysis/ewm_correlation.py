# Rolling Average Sentiment
# In check_hypothesis, we were comparing the stock price at a given day x
# versus the twitter sentiment at day x-n, where n is a hyperparameter.

# For rolling averages, we want to compare the stock price at a given day x
# versus the twitter sentiment average from day 0 to day x-n. 
# The aim is to quantify sentiment over a period against stock price some time after
# that period. 

# A lot of the work is very similar to the check_hypothesis file.

import pandas as pd 
import numpy as np 
import twint
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.express as px

from check_hypothesis.py import base_processor

class ewm_processor(base_processor):
    def __init__(self):
        super().__init__()

    def get_correlation(self, data, stock_data, start_date, day_difference):
        """
        get_correlation: Function which returns correlation between time-shifted num of tweets and the 
        stock price for a given day. What we do here is that we shift the day of creation of tweets by 
        day_difference so that we compare past counts of tweets with a specific day's stock price. 
        It's a very hack-y function but this is EDA anyway, not performance coding.

        In the cumulative mean class, we convert the daily number of tweets to a cumulative exponential
        average (that gives bias to the nearer data point). This is supposed to smooth the data and
        present us with sentiment information over a period, rather than a single day.

        args:
            data (pd.DataFrame): DataFrame containing tweet information (what's really needed is the
                                datetime value)
            stock (pd.DataFrame): Stock Ticker data, from yfinance
            start_date (string): Date to get stock prices from
            day_difference (int): The amount of the time-shift (in days)

        out:
            (float): The value of the correlation
        """
        # Reduces stock DataFrame to only days required
        stock_data = stock_data[stock_data.index >= start_date]
        x = stock_data['Close']

        # Creates deep copy of tweets and shifts tweet dates by day_difference
        data_f = data[['date','id']].copy(deep=True)
        data_f['datetime'] = data_f['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) + timedelta(days=day_difference)

        # Strips datetime object into required date format
        data_f['date'] = data_f['datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))

        # Groups by date and gets averages for different values
        num_tweets = data_f.groupby('date')['id'].count()

        # Cumulative exponential mean
        num_tweets['ewm'] = num_tweets['id'].ewm().mean()
        num_tweets = num_tweets.to_frame().join(x, how='inner')

        # Calculates Pearson's correlation coefficient
        return np.corrcoef(num_tweets['ewm'], num_tweets['Close'])[0, 1]

if __name__ == "__main__":
    stock_tickers = {'NVDA' : 'Nvidia'}
    start_date = datetime.strptime('2020-10-01', '%Y-%m-%d')
    processor = ewm_processor()
    output = processor.process_all_stocks(stock_tickers, start_date, 50, 0, 60)

    output = pd.DataFrame.from_dict(output)
    output.to_csv('tweet_correlations_ewm.csv')



