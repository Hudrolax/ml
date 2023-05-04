from .tester_base_class import TesterBaseClass
from .data_classes import Actions
import numpy as np


class BBTester(TesterBaseClass):
    """
        Bolliger bands strategy. Open and close orders by bollinger bands and action
        Open order if action > x
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: float 
                risk: float32 (percent volume of balance),
        """

        reward = 0
        tick = self.tick
        bid = tick['open']
        actions = action['action']

        # close open orders
        tick_pnl = 0
        for order in self.open_orders:
            if order.type == Actions.Buy:
                # Buy
                if bid >= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if bid <= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
        
        if tick_pnl > 0:
            reward += 10
        elif tick_pnl < 0:
            reward -= 20

        # Open orders.
        if len(self.open_orders) == 0:

            if bid <= tick['bb_lower']:
                if actions[0] > 0.5:
                    risk = action['risk'] * (1 + actions[0] - 0.5)
                    self.open_order(
                        order_type=Actions.Buy,
                        vol=self.balance * risk,
                    )
                else:
                    reward -= 1

            elif bid >= tick['bb_upper']:
                if actions[1] > 0.5:
                    risk = action['risk'] * (1 + actions[0] - 0.5)
                    self.open_order(
                        order_type=Actions.Sell,
                        vol=self.balance * risk,
                    )
                else:
                    reward -= 1
        return reward


class BBFreeActionTester(TesterBaseClass):
    """
        Open and close orders by bollinger bands without any policy.
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: not use,
                risk: float32 (percent volume of balance),
        """

        tick = self.tick
        price = tick['open']

        # close open orders
        tick_pnl = 0
        for order in self.open_orders:
            if order.type == Actions.Buy:
                # Buy
                if price >= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if price <= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)

        # Open orders.
        if len(self.open_orders) == 0:
            if price <= tick['bb_lower']:
                self.open_order(
                    order_type=Actions.Buy,
                    vol=self.balance * action['risk'],
                )
            elif price >= tick['bb_upper']:
                self.open_order(
                    order_type=Actions.Sell,
                    vol=self.balance * action['risk'],
                )

        return tick_pnl

class BBTesterSortino(BBTester):
    def on_tick(self, *args, **kwargs) -> ...:
        self._tick = self.klines.iloc[self.n_tick]
        self.n_tick += 1

        # check Done
        self.check_done()

        # Calculate P&L
        general_pnl = 0
        for order in self.open_orders:
            order.pnl, _ = self._order_pnl(order)
            general_pnl += order.pnl
        self.equity.append(self.balance + general_pnl)

        # margin call if depo <= 20%
        if self.balance <= self.start_depo * 0.2:
            self.margin_call = True
        
        self._on_tick(*args, **kwargs)

        reward = 0
        if self.done and self.do_render:
            self.render()

        if self.done:
            reward = self.info()['sortino']

        return reward
