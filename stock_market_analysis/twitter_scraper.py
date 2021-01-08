import twint

c = twint.Config()

c.Search = "TSLA"
c.Store_csv = True
c.Min_likes = 100
c.Lang = "en"
c.Output = "stock_market_analysis/tesla_2016.csv"
c.Since = '2016-01-01'

twint.run.Search(c)

