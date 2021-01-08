# This is a python file intended to prove our hypothesis:
# Do Twitter sentiment and number of tweets have any impact, at all, on the stock price of companies?

# We will aim to prove this hypothesis by scraping tweets for many companies and comparing them to future
# stock prices for multiple companies and see if correlations are high between the tweets.
# However, correlation does not mean causation.

# To imply causation, we need a stronger data proof than just correlation.

# Does causation matter? If we can imply correlation we can concur that tweet sentiment is a valuable
# tool for analyzing stock trends. 

import pandas as pd 
import numpy as np
import twint
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.express as px

def get_correlation(data, stock_data, start_date, day_difference):
    """
    get_correlation: Function which returns correlation between time-shifted num of tweets and the 
    stock price for a given day. What we do here is that we shift the day of creation of tweets by 
    day_difference so that we compare past counts of tweets with a specific day's stock price. 
    It's a very hack-y function but this is EDA anyway, not performance coding.

    args:
        data (pd.DataFrame): DataFrame containing tweet information (what's really needed is the
                             datetime value)
        stock (pd.DataFrame): Stock Ticker data, from yfinance
        start_date (string): Date to get stock prices from
        day_difference (int): The amount of the time-shift (in days)
    """
    # Reduces stock DataFrame to only days required
    stock_data['datetime'] = stock_data.index.apply(lambda y: datetime.strptime(y, '%Y-%m-%d'))
    stock_data = stock_data[stock_data['datetime'] >= start_date]
    x = stock_data['Close']

    # Creates deep copy of tweets and shifts tweet dates by day_difference
    data_f = data[['datetime', 'id']].copy(deep=True)
    data_f['datetime'] = data_f['datetime'] + timedelta(days=day_difference)

    # Strips datetime object into required date format
    data_f['date'] = data_f['datetime'].apply(lambda x: x.strftime("%Y-%m-%d"))

    # Groups by date and gets averages for different values
    num_tweets = data_f.groupby('date')['id'].count()
    num_tweets = num_tweets.to_frame().join(x, how='inner')

    # Calculates Pearson's correlation coefficient
    return np.corrcoef(num_tweets['id'], num_tweets['Close'])[0, 1]

def process_all_stocks(stock_tickers, start_date, min_likes, shift_range):
    # Initialize 
    c = twint.Config()
    output = {}

    # Check if there are stock acronyms (?) in the stock_tickers list
    assert(len(stock_tickers) > 0)

    # Check - if the amount of shift is greater than the number of days
    # we are taking tweets for, we have a problem
    assert(shift_range < (datetime.now() - start_date).days)

    for stock in stock_tickers:
        
        # Define tweet search parameters
        c.Search = stock
        c.Pandas = True
        c.Min_likes = min_likes
        c.Lang = "en"
        c.Since = start_date.strftime('%Y-%m-%d')

        # Scrape twitter data and store into DataFrame
        twint.run.Search(c)
        data = twint.storage.panda.Tweets_df

        # Scrape stock ticker data
        stock_data = yf.download(stock, interval='1d', start=start_date)

        # Get correlation between time-shifted tweet data and stock ticker data
        out = []
        for i in range(0, shift_range):
            start_shifted = start_date + timedelta(days=i)
            out.append(get_correlation(data, stock_data, start_shifted, i))

        output[stock] = out

    return output



    
if __name__ == "__main__":
    stock_tickers = ['TSLA']
    start_date = datetime.strptime('2020-12-31')
    output = process_all_stocks(stock_tickers, start_date, 100, 181)

    output = pd.DataFrame.from_dict(output)
    output.to_csv('tweet_correlations.csv')

    

