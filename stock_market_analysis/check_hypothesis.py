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
import yfinance
from textblob import TextBlob
from datetime import datetime, timedelta

stock_tickers = ['TSLA']
c = twint.Config()

for stock in stock_tickers:
    
    c.Search = stock
    c.Pandas = True
    c.Min_likes = 100
    c.Lang = "en"
    c.Since = '2020-12-12'

    twint.run.Search(c)
    data = twint.storage.panda.Tweets_df

    print(data.head())


