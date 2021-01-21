# This is a python file intended to prove our hypothesis:
# Do Twitter sentiment and number of tweets have any impact, at all, on the stock price of companies?

# We will aim to prove this hypothesis by scraping tweets for many companies and comparing them to future
# stock prices for multiple companies and see if correlations are high between the tweets.
# However, correlation does not mean causation.

# To imply causation, we need a stronger data proof than just correlation.

# Does causation matter? If we can imply correlation we can concur that tweet sentiment is a valuable
# tool for analyzing stock trends. 

# <TODO> 
# Major problem noticed. Market is closed on weekends. We do an inner join to match stocks and 
# twitter feeds, which is obviously getting rid of the weekend tweet data. 
# I'm not sure how to fix this.
# </TODO>

import pandas as pd 
import numpy as np
import twint
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.express as px

# Helper functions
def scrape_twitter(c, search_value, min_likes, min_retweets, start_date, end_date=None):
    """
    scrape_twitter: A simple twint-based function that scrapes twitter based on search
                    parameters.
    args:
        c (twint.Config) : A Twint Config object
        search_value (string) : The string to base the search on
        min_likes, min_retweets (int) : Minimum likes and retweets
        start_date (datetime.datetime) : The date to start search
    """
    # Define tweet search parameters
    c.Search = search_value
    c.Pandas = True
    c.Min_likes = min_likes
    c.Min_retweets = min_retweets 
    c.Lang = "en"
    c.Since = start_date.strftime('%Y-%m-%d')

    # Scrape twitter data and store into DataFrame
    twint.run.Search(c)
    return twint.storage.panda.Tweets_df


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
    num_tweets = num_tweets.to_frame().join(x, how='inner')

    #fig = px.scatter(num_tweets, 'id', 'Close')
    #fig.show()

    # Calculates Pearson's correlation coefficient
    return np.corrcoef(num_tweets['id'], num_tweets['Close'])[0, 1]

def process_all_stocks(stock_tickers, start_date, min_likes, min_retweets, shift_range):
    """
    process_all_stocks: Over-arching function that downloads tweets, downloads stock prices,
    and produces correlation for all days in the shift range. This is going to be quite a slow
    process, depending on the size of data required to download and process. 

    args:
        stock_tickers (list): List of all stocks to be analyzed by the process
        start_date (datetime.datetime object): Starting date from which tweets and stock
                                               ticker data is to be downloaded and processed
        min_likes (int): Twint search requirement - defines the minimum number of likes 
                         required for a tweet to be picked up by the scraper
        shift_range (int): Range of days - [0, shift_range] - the time-shifting is to be
                           done for.
    
    output:
        (dict): A dictionary containing mappings of stock acronyms to their correlation lists.
    """
    # Initialize 
    c = twint.Config()
    output = {}

    # Check if there are stock acronyms (?) in the stock_tickers list
    assert(len(stock_tickers) > 0)

    # Check - if the amount of shift is greater than the number of days
    # we are taking tweets for, we have a problem
    assert(shift_range < (datetime.now() - start_date).days)

    # Check - start date has to be less than or equal to the current date
    assert(start_date <= datetime.datetime.now())

    for key, value in stock_tickers.items():
        
        # Define tweet search parameters and scrape 
        data = scrape_twitter(c, value, min_likes, min_retweets, start_date)

        # Calculate sentiment
        data['tweet_sentiment'] = data['tweet'].apply(lambda x: TextBlob(x).sentiment)
        data['tweet_sentiment_polarity'] = data['tweet_sentiment'].apply(lambda x: x.polarity)
        data['tweet_sentiment_subjectivity'] = data['tweet_sentiment'].apply(lambda x: x.subjectivity)
        data.drop(columns=['tweet_sentiment'], inplace=True)

        # Keep ONLY positive tweets
        data = data[data['tweet_sentiment_polarity'] > 0]

        # Scrape stock ticker data
        stock_data = yf.download(key, interval='1d', start=start_date)

        # Get correlation between time-shifted tweet data and stock ticker data
        out = []
        for i in range(0, shift_range):
            start_shifted = start_date + timedelta(days=i)
            out.append(get_correlation(data, stock_data, start_shifted, i))

        output[key] = out

    return output


class base_processor:
    def __init__(self):
        self.c = twint.Config()

    """
    scrape_twitter: A simple twint-based function that scrapes twitter based on search
                    parameters.
    args:
        c (twint.Config) : A Twint Config object
        search_value (string) : The string to base the search on
        min_likes, min_retweets (int) : Minimum likes and retweets
        start_date (datetime.datetime) : The date to start search
    """
    def scrape_twitter(self, search_value, min_likes, min_retweets, start_date, end_date=None):

        # Define tweet search parameters
        self.c.Search = search_value
        self.c.Pandas = True
        self.c.Min_likes = min_likes
        self.c.Min_retweets = min_retweets 
        self.c.Lang = "en"
        self.c.Since = start_date.strftime('%Y-%m-%d')

        # Scrape twitter data and store into DataFrame
        twint.run.Search(self.c)
        return twint.storage.panda.Tweets_df

    def get_correlation(self, data, stock_data, start_date, day_difference):
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
        num_tweets = num_tweets.to_frame().join(x, how='inner')

        #fig = px.scatter(num_tweets, 'id', 'Close')
        #fig.show()

        # Calculates Pearson's correlation coefficient
        return np.corrcoef(num_tweets['id'], num_tweets['Close'])[0, 1]

    """
    process_all_stocks: Over-arching function that downloads tweets, downloads stock prices,
    and produces correlation for all days in the shift range. This is going to be quite a slow
    process, depending on the size of data required to download and process. 

    args:
        stock_tickers (list): List of all stocks to be analyzed by the process
        start_date (datetime.datetime object): Starting date from which tweets and stock
                                            ticker data is to be downloaded and processed
        min_likes (int): Twint search requirement - defines the minimum number of likes 
                        required for a tweet to be picked up by the scraper
        shift_range (int): Range of days - [0, shift_range] - the time-shifting is to be
                        done for.
    
    output:
        (dict): A dictionary containing mappings of stock acronyms to their correlation lists.
    """
    def process_all_stocks(self, stock_tickers, start_date, min_likes, min_retweets, shift_range):
        output = {}

        # Check if there are stock acronyms (?) in the stock_tickers list
        assert(len(stock_tickers) > 0)

        # Check - if the amount of shift is greater than the number of days
        # we are taking tweets for, we have a problem
        assert(shift_range < (datetime.now() - start_date).days)

        # Check - start date has to be less than or equal to the current date
        assert(start_date <= datetime.now())

        for key, value in stock_tickers.items():
            
            # Define tweet search parameters and scrape 
            data = self.scrape_twitter(value, min_likes, min_retweets, start_date)

            # Calculate sentiment
            data['tweet_sentiment'] = data['tweet'].apply(lambda x: TextBlob(x).sentiment)
            data['tweet_sentiment_polarity'] = data['tweet_sentiment'].apply(lambda x: x.polarity)
            data['tweet_sentiment_subjectivity'] = data['tweet_sentiment'].apply(lambda x: x.subjectivity)
            data.drop(columns=['tweet_sentiment'], inplace=True)

            # Keep ONLY positive tweets
            data = data[data['tweet_sentiment_polarity'] > 0]

            # Scrape stock ticker data
            stock_data = yf.download(key, interval='1d', start=start_date)

            # Get correlation between time-shifted tweet data and stock ticker data
            out = []
            for i in range(0, shift_range):
                start_shifted = start_date + timedelta(days=i)
                out.append(get_correlation(data, stock_data, start_shifted, i))

            output[key] = out

        return output 


if __name__ == "__main__":
    stock_tickers = {'NVDA' : 'Nvidia'}
    start_date = datetime.strptime('2020-10-01', '%Y-%m-%d')
    bp = base_processor()
    output = bp.process_all_stocks(stock_tickers, start_date, 50, 0, 60)

    output = pd.DataFrame.from_dict(output)
    output.to_csv('tweet_correlations.csv')

    

