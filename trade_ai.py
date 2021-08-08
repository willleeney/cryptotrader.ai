from coain.dataset.cryptodownload import CryptoDataDownload
from coain.dataset.createfeatures import create_basic_features, rsi, macd
from coain.renderer.history_plot import plot_df
from coain.renderer.default import MyPlotlyTradingChart
from coain.TheScheme.buysellhold import BuySellHold, PBR, SharpeRatio, MySimpleOrders

from tensortrade.feed.core import Stream, DataFeed
import tensortrade.env.default as default
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import BTC, ETH
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.wallets import Wallet, Portfolio

from tensortrade.agents import DQNAgent


import pandas as pd
import plotly.graph_objects as go


def run():

    exchange_name = 'binance'
    quote_symbol = 'ETH'
    base_symbol = 'USDT'
    timeframe = 'h'

    save = False
    view = False
    create = True

    CryptoData = CryptoDataDownload()

    price_history, filename = CryptoData.fetch(exchange_name, base_symbol, quote_symbol, timeframe)
    price_history = price_history.rename(columns={'date': 'time'})

    csv_name = '{}.csv'.format(filename)
    if save:
        price_history.to_csv(csv_name)

    if view:
        plot_df(price_history, filename, quote_symbol, base_symbol)

    price_history = price_history[-500:]
    if create:
        # creates trading features
        tidy_price_histroy = create_basic_features(price_history)


    # creates a list of streams for use in RL environment
    data = tidy_price_histroy

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    close_price = features[3]

    features += [rsi(close_price, period=20).rename("rsi")]
    features += [macd(close_price, fast=10, slow=50, signal=5).rename("macd")]

    feed = DataFeed(features)
    feed.compile()

    # creates the exchange in which orders are created
    binance = Exchange("binance", service=execute_order)(
        close_price.rename("USDT-ETH")
    )

    # create the instrument
    USDT = Instrument("USDT", 3, "U.S. Dollar Tender")
    # define the starting amounts
    cash = Wallet(binance, 1000 * USDT)
    asset = Wallet(binance, 0 * ETH)

    portfolio = Portfolio(USDT, [
        cash,
        asset
    ])

    reward_scheme = PBR(price=close_price)

    action_scheme = BuySellHold(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    #reward_scheme = SharpeRatio()

    #action_scheme = MySimpleOrders(trade_sizes=2)

    renderer_feed = DataFeed([
        Stream.source(list(data["time"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume eth"]), dtype="float").rename("volume")
    ])

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=MyPlotlyTradingChart(),
        window_size=20
    )

    #env.observer.feed.next()

    agent = DQNAgent(env)
    agent.train(n_steps=500, n_episodes=10, render_interval=None, save_path="agents/")





if __name__ == "__main__":
    run()