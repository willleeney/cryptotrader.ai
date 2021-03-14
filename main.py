import random
import pandas as pd

import tensortrade.stochastic as sp
import tensortrade.env.default as default

from tensortrade.feed.core import Stream, DataFeed
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.agents import DQNAgent


def run():

    cdd = CryptoDataDownload()
    bitfinex_btc = cdd.fetch("Bitfinex", "USD", "BTC", "1h")

    bitfinex = Exchange("bitfinex", service=execute_order)(
        Stream.source(list(bitfinex_btc['close']), dtype="float").rename("USD-BTC"),

    )

    portfolio = Portfolio(USD, [
        Wallet(bitfinex, 10000 * USD),
        Wallet(bitfinex, 10 * BTC)
    ])

    features = []
    data = bitfinex_btc
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")

    features = [
        cp.log().diff().rename("lr"),
        rsi(cp, period=20).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd")
    ]

    feed = DataFeed(features)
    feed.compile()

    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])

    chart_renderer = default.renderers.MatplotlibTradingChart(
        display=True  # show the chart on screen (default
          # save the chart to an HTML file
    )

    env = default.create(
        portfolio=portfolio,
        action_scheme="managed-risk",
        reward_scheme="risk-adjusted",
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=['screen-log'],
        window_size=20
    )

    agent = DQNAgent(env)
    agent.train(n_steps=24800, n_episodes=10, save_path="agents/", render_interval=100)


    return

def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)

def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal

if __name__ == "__main__":
    run()