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
                action: float (0-'Pass', 1-'Buy'),
                risk: float32 (percent volume of balance),
        """

        reward = 0
        tick = self.tick
        bid = tick['open']
        actions = action['action']

        if not isinstance(actions, np.ndarray):
            raise ValueError(f'actions must be a ndarray. Actions is {actions}')
        
        assert isinstance(actions[0], np.float32)
        assert isinstance(actions[1], np.float32)

        # get open orders
        open_orders = self.get_open_orders()

        # close open orders
        tick_pnl = 0
        for order in open_orders:
            if order.type == Actions.Buy:
                # Buy
                if bid >= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if bid <= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
        
        if tick_pnl > 0:
            reward += 15
        elif tick_pnl < 0:
            reward -= 10

        # get open orders again
        open_orders = self.get_open_orders()

        # Open orders.
        if len(open_orders) == 0:

            if bid <= tick['bb_lower']:
                if actions[0] > 0.5:
                    risk = action['risk'] * (1 + actions[0] - 0.5)
                    self.open_order(
                        order_type=Actions.Buy,
                        vol=self.balance * risk * (len(open_orders) + 1),
                    )
                else:
                    reward -= 1

            elif bid >= tick['bb_upper']:
                if actions[0] > 0.5:
                    risk = action['risk'] * (1 + actions[0] - 0.5)
                    self.open_order(
                        order_type=Actions.Sell,
                        vol=self.balance * risk * (len(open_orders) + 1),
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
        bid = tick['open']

        # get open orders
        open_orders = self.get_open_orders()

        # close open orders
        tick_pnl = 0
        for order in open_orders:
            if order.type == Actions.Buy:
                # Buy
                if bid >= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if bid <= tick['bb_middle'] or self.done:
                    tick_pnl += self.close_order(order)

        # get open orders again
        open_orders = self.get_open_orders()

        # Open orders.
        if len(open_orders) == 0:
            if bid <= tick['bb_lower']:
                self.open_order(
                    order_type=Actions.Buy,
                    vol=self.balance * action['risk'] * (len(open_orders) + 1),
                )
            elif bid >= tick['bb_upper']:
                self.open_order(
                    order_type=Actions.Sell,
                    vol=self.balance * action['risk'] * (len(open_orders) + 1),
                )

        return tick_pnl
