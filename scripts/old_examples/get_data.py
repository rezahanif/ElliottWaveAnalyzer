import yfinance as yf

from src.waveconf.legacy_models.helpers import convert_yf_data

df = yf.download(tickers='AAPL',
                 interval="1d",
                 start="2022-12-01")

convert_yf_data(df).to_csv(r'data/raw/aapl_1d_2020.csv', sep=",", index=False)
