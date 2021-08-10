import pandas as pd
import numpy as np
from gym.spaces import Discrete

from tensortrade.env.default.actions import TensorTradeActionScheme
from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed

from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.wallets import Portfolio


import logging
from abc import abstractmethod
from itertools import product
from typing import Union, List, Any

from gym.spaces import Space, Discrete

from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (
    Broker,
    Order,
    OrderListener,
    OrderSpec,
    proportion_order,
    risk_managed_order,
    TradeSide,
    TradeType
)

class BuySellHold(TensorTradeActionScheme):

    registered_name = "buysellhold"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0


class PBR(TensorTradeRewardScheme):

    registered_name = "pbr"

    def __init__(self, price: 'Stream'):
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (r * position).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        return self.feed.next()["reward"]

    def reset(self):
        self.position = -1
        self.feed.reset()


class SharpeRatio(TensorTradeRewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth,
       while penalizing more volatile strategies.
       Parameters
       ----------
       return_algorithm : {'sharpe', 'sortino'}, Default 'sharpe'.
           The risk-adjusted return metric to use.
       risk_free_rate : float, Default 0.
           The risk free rate of returns to use for calculating metrics.
       target_returns : float, Default 0
           The target returns per period for use in calculating the sortino ratio.
       window_size : int
           The size of the look back window for computing the reward.
       """


    def __init__(self,
                 risk_free_rate: float = 0.,
                 target_returns: float = 0.,
                 window_size: int = 1) -> None:

        self._risk_free_rate = self.default('risk_free_rate', risk_free_rate)
        self._target_returns = self.default('target_returns', target_returns)
        self._window_size = self.default('window_size', window_size)


    def _sharpe_ratio(self, returns: 'pd.Series') -> float:
        """Computes the sharpe ratio for a given series of a returns.
        Parameters
        ----------
        returns : `pd.Series`
            The returns for the `portfolio`.
        Returns
        -------
        float
            The sharpe ratio for the given series of a `returns`.
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (np.mean(returns) - self._risk_free_rate + 1e-9) / (np.std(returns) + 1e-9)


    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Computes the reward corresponding to the selected risk-adjusted return metric.
        Parameters
        ----------
        portfolio : `Portfolio`
            The current portfolio being used by the environment.
        Returns
        -------
        float
            The reward corresponding to the selected risk-adjusted return metric.
        """
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()][-(self._window_size + 1):]
        returns = pd.Series(net_worths).pct_change().dropna()
        risk_adjusted_return = self._sharpe_ratio(returns)
        return risk_adjusted_return


class MySimpleOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes.
    Parameters
    ----------
    criteria : List[OrderCriteria]
        A list of order criteria to select from when submitting an order.
        (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : TradeType
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(self,
                 criteria: 'Union[List[OrderCriteria], OrderCriteria]' = None,
                 trade_sizes: 'Union[List[float], int]' = 10,
                 durations: 'Union[List[int], int]' = None,
                 trade_type: 'TradeType' = TradeType.MARKET,
                 order_listener: 'OrderListener' = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        criteria = self.default('criteria', criteria)
        self.criteria = criteria if isinstance(criteria, list) else [criteria]

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = product(
                self.criteria,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self,
                   action: int,
                   portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        order = Order(
            step=self.clock.step,
            side=side,
            trade_type=self._trade_type,
            exchange_pair=ep,
            price=ep.price,
            quantity=quantity,
            criteria=criteria,
            end=self.clock.step + duration if duration else None,
            portfolio=portfolio
        )

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]