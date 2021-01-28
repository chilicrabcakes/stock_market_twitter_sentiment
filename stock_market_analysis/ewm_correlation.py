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

from check_hypothesis import base_processor

class ewm_processor(base_processor):
    def __init__(self):
        super().__init__()

    def get_correlation(self, data, stock_data, start_date, day_difference):
        """
        get_correlation: Function which returns correlation between time-shifted num of tweets and the 
        stock price for a given day. What we do here is that we shift the day of creation of tweets by 
        day_difference so that we compare past counts of tweets with a specific day's stock price. 
        It's a very hack-y function but this is EDA anyway, not performance coding.

        (ewm_processor changes)

        Instead of correlation with count, we have added a weighted average of num of likes and sentiment.
        This is a different way of viewing information, but we do need to include the number of tweets 
        somehow in the future. 

        In the inherited EWM class, we convert the daily number of tweets to a cumulative exponential 
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
        data_f = data[['date','tweet_sentiment_polarity']].copy(deep=True)
        data_f['datetime'] = data_f['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) + timedelta(days=day_difference)

        # Strips datetime object into required date format
        data_f['date'] = data_f['datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))

        # Groups by date and gets weighted count by sentiment
        # num_tweets = data_f.groupby('date')['id'].count()
        data_f['count'] = 1
        #weighted_sentiment = data_f.groupby('date').apply(lambda x: np.average(x['nlikes'], weights=x['tweet_sentiment_polarity']))
        weighted_sentiment = data_f.groupby('date').apply(lambda x: x['count'].dot(x['tweet_sentiment_polarity']))

        # Exponential weighted mean of the weighted sentiment
        weighted_sentiment = weighted_sentiment.ewm(span=40, adjust=False).mean()
        weighted_sentiment = weighted_sentiment.to_frame().join(x, how='inner')

        # Calculates Pearson's correlation coefficient
        return np.corrcoef(weighted_sentiment.iloc[:, 0], weighted_sentiment.iloc[:, 1])[0, 1]

if __name__ == "__main__":

    stock_tickers = {'GME' : 'GameStop'}
    start_date = datetime.strptime('2020-11-01', '%Y-%m-%d')
    processor = ewm_processor()
    output = processor.process_all_stocks(stock_tickers, start_date, 0, 0, 20)

    output = pd.DataFrame.from_dict(output)
    output.to_csv('tweet_correlations_ewm.csv')
    fig = px.line(output, x=output.index, y=['GME'], title='Weighted Correlation between past tweets and stock price')
    fig.update_xaxes(title_text='Days past',rangeslider_visible=True)
    fig.update_yaxes(title_text='Correlation')
    fig.show()



