from .tester_base_class import TesterBaseClass
from .data_classes import Actions


class GymFuturesTester(TesterBaseClass):

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: int (0-'Sell', 1-'Buy', 2-'Pass'),
                risk: float32 (percent volume of balance),
        """
        assert action.get('action') is not None
        assert action.get('risk') is not None

        # get open orders
        open_orders = self.get_open_orders()

        # close open orders
        tick_pnl = 0
        for order in open_orders:
            if order.type == Actions.Buy:
                # Buy
                if self.tick['open'] >= order.TP or self.tick['open'] <= order.SL or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if self.tick['open'] <= order.TP or self.tick['open'] >= order.SL or self.done:
                    tick_pnl += self.close_order(order)

        if action['action'] == Actions.Pass:
            # pass
            return tick_pnl

        # get open orders again
        open_orders = self.get_open_orders()

        # Open orders.
        if not open_orders:
            _tp = self.tick['open'] * self.TP/100
            _sl = self.tick['open'] * self.SL/100
            if action['action'] == Actions.Buy:
                _tp = self.tick['open'] + _tp
                _sl = self.tick['open'] - _sl
            else:
                _tp = self.tick['open'] - _tp
                _sl = self.tick['open'] + _sl

            self.open_order(
                order_type=Actions(action['action']),
                vol=self.balance * action['risk'],
                TP=_tp,
                SL=_sl,
            )

        return tick_pnl


class BBFutureTester(TesterBaseClass):
    """
        Bolliger bands strategy
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: int (0-'Sell', 1-'Buy', 2-'Pass'),
                risk: float32 (percent volume of balance),
        """
        assert action.get('action') is not None
        assert action.get('risk') is not None

        tick = self.tick
        bid = tick['open']
        act = action.get('action')

        # get open orders
        open_orders = self.get_open_orders()

        # close open orders
        tick_pnl = 0
        for order in open_orders:
            if order.type == Actions.Buy:
                # Buy
                if bid >= tick['bb_middle'] or self.done or act == 0:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if bid <= tick['bb_middle'] or self.done or act == 1:
                    tick_pnl += self.close_order(order)

        # get open orders again
        open_orders = self.get_open_orders()

        gap = (tick['bb_upper'] - tick['bb_lower']) / 4
        lower_buy = float('inf')
        highest_sell = float('-inf')
        for order in open_orders:
            if order.open < lower_buy:
                lower_buy = order.open
            if order.open > highest_sell:
                highest_sell = order.open

        # Open orders.
        if len(open_orders) < 7:
            if bid <= tick['bb_lower'] and bid < lower_buy - gap and act == 1:
                self.open_order(
                    order_type=Actions.Buy,
                    vol=self.balance * action['risk'] * (len(open_orders) + 1),
                )
            elif bid >= tick['bb_upper'] and bid > highest_sell + gap and act == 0:
                self.open_order(
                    order_type=Actions.Sell,
                    vol=self.balance * action['risk'] * (len(open_orders) + 1),
                )

        return tick_pnl


class BBFutureTester2(TesterBaseClass):
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

        tick = self.tick
        bid = tick['open']
        act = action['action']

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
        if len(open_orders) == 0 and act > 0.5:
            risk = action['risk'] * (1 + act - 0.5)

            if bid <= tick['bb_lower']:
                self.open_order(
                    order_type=Actions.Buy,
                    vol=self.balance * risk * (len(open_orders) + 1),
                )
            elif bid >= tick['bb_upper']:
                self.open_order(
                    order_type=Actions.Sell,
                    vol=self.balance * risk * (len(open_orders) + 1),
                )

        return tick_pnl


class BBFutureTester3(TesterBaseClass):
    """
        This tester use BB strategy and returns momental reward.
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: float (0-'Sell', 1-'Buy'),
                risk: float32 (percent volume of balance),
        """

        tick = self.tick
        price = tick['open']
        open_action = action['action']
        pnl_percent = 0

        def calculate_pnl(open_price, close_price, action):
            fee = 0.0004
            vol = 100
            fee_sum = vol * fee + vol / open_price * close_price * fee
            pnl = vol / open_price * close_price - vol - fee_sum
            pnl_percent = pnl / vol
            return pnl_percent if action == 1 else -pnl_percent

        if open_action > 0.8 and price < tick['bb_lower']:
            # buy strategy
            df = self.klines.iloc[self.n_tick:]
            mask = df['close'] >= df['bb_upper']
            try:
                close = df.loc[mask, 'close'].iloc[0]
            except:
                close = tick['close']
            pnl_percent = calculate_pnl(price, close, 1)

            # print(f"action {open_action}. time {tick['date']} open {price} close {close}, pnl {pnl_percent}")
        elif open_action < 0.2 and price > tick['bb_upper']:
            # sell strategy
            df = self.klines.iloc[self.n_tick:]
            mask = df['close'] <= df['bb_lower']
            try:
                close = df.loc[mask, 'close'].iloc[0]
            except:
                close = tick['close']
            pnl_percent = calculate_pnl(price, close, 0)
            # print(f"action {open_action}. time {tick['date']} open {price} close {close}, pnl {pnl_percent}")

        return pnl_percent


class BBFreeActionTester(TesterBaseClass):
    """
        Open and close orders by bollinger bands without policy.
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: float not use,
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
