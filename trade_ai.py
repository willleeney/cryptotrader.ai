from coain.dataset.cryptodownload import CryptoDataDownload
from coain.dataset.createfeatures import create_basic_features
from coain.renderer.history_plot import plot_df

import pandas as pd
import plotly.graph_objects as go


def run():

    exchange_name = 'binance'
    quote_symbol = 'XLM'
    base_symbol = 'USDT'
    timeframe = 'm'

    save = False
    view = False
    create = True

    CryptoData = CryptoDataDownload()

    price_history, filename = CryptoData.fetch(exchange_name, base_symbol, quote_symbol, timeframe)
    price_history = price_history.rename(columns={'date': 'time'})

    csv_name = '{}.csv'.format(filename)
    if save:
        price_history.to_csv(csv_name)

    price_history = price_history[-500:]
    if view:

        plot_df(price_history, filename, quote_symbol, base_symbol)

    if create:
        tidy_price_histroy = create_basic_features(price_history)

        print(tidy_price_histroy.columns)





if __name__ == "__main__":
    run()