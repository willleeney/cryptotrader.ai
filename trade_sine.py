from coain.dataset.cryptodownload import CryptoDataDownload
from coain.dataset.createfeatures import create_basic_features, rsi, macd
from coain.renderer.history_plot import plot_df
from coain.TheScheme.buysellhold import BuySellHold, PBR, SharpeRatio, MySimpleOrders
from coain.renderer.default import PositionChangeChart

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
import numpy as np

USDT = Instrument("USDT", 2, "U.S. Dollar Tender")
TTT = Instrument("TTT", 8, "TensorTrade Test")


def run():

    # create some fake data
    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1001)
    y = 50 * np.sin(3 * x) + 100

    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    price = Stream.source(y, dtype="float").rename("USD-TTT")

    # create exchange
    binance = Exchange("binance", service=execute_order)(
        price.rename("USDT-TTT")
    )

    # define the starting amounts
    cash = Wallet(binance, 1000 * USDT)
    asset = Wallet(binance, 0 * TTT)

    portfolio = Portfolio(USDT, [
        cash,
        asset
    ])

    # define the features
    feed = DataFeed([
        price,
        price.rolling(window=10).mean().rename("fast"),
        price.rolling(window=50).mean().rename("medium"),
        price.rolling(window=100).mean().rename("slow"),
        price.log().diff().fillna(0).rename("lr")
    ])
    feed.compile()

    reward_scheme = SharpeRatio()

    action_scheme = MySimpleOrders(trade_sizes=4)

    renderer_feed = DataFeed([
        Stream.source(y, dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.actions, dtype="float").rename("action")
    ])

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=PositionChangeChart(),
        window_size=20
    )

    agent = DQNAgent(env)
    agent.train(n_steps=1000, n_episodes=10, render_interval=None, save_path="agents/")


if __name__ == "__main__":
    run()