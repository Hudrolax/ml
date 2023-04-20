import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_classes import Order, Actions
import numpy as np
from matplotlib.colors import ColorConverter


def interpolate_colors(color1, color2, step):
    """
    Возвращает список цветов с плавным переходом от цвета color1 к цвету color2.
    Число элементов списка равно step.

    Аргументы:
    color1 (tuple): цвет 1 в формате RGB.
    color2 (tuple): цвет 2 в формате RGB.
    step (int): число элементов списка.

    Возвращает:
    Список цветов в формате RGB.
    """
    # Преобразование цветов в объекты Color
    color1_rgb = np.array(color1) / 255.0
    color2_rgb = np.array(color2) / 255.0
    color1_obj = ColorConverter().to_rgb(color1_rgb)
    color2_obj = ColorConverter().to_rgb(color2_rgb)

    # Генерация списка промежуточных цветов
    r = np.linspace(color1_obj[0], color2_obj[0], step)
    g = np.linspace(color1_obj[1], color2_obj[1], step)
    b = np.linspace(color1_obj[2], color2_obj[2], step)

    # Преобразование промежуточных цветов в формат RGB
    colors_rgb = np.transpose(np.vstack((r, g, b))) * 255.0
    colors_rgb = colors_rgb.astype(int)

    # Преобразование списка цветов в список кортежей
    # colors_list = [tuple(colors_rgb[i]) for i in range(step)]
    colors_list = [
        f"rgb({colors_rgb[i][0]},{colors_rgb[i][1]},{colors_rgb[i][2]})" for i in range(step)]

    return colors_list


def order_annotations(
        klines: pd.DataFrame,
        orders: list[Order],
        text_annotation: str,
        buy_color: str = '#017005',
        sell_color: str = 'rgb(220,0,0)',
) -> list[dict]:
    """Write order text annotations"""
    colors = {'buy': buy_color, 'sell': sell_color}
    result = []
    for order in orders:
        text = ''
        if text_annotation:
            if text_annotation == 'pnl':
                text = round(order.pnl, 2)
            elif text_annotation == 'amount':
                text = order.vol
            elif text_annotation == 'pnl+amount':
                text = f'{round(order.pnl, 2)} ({order.vol})'

        annotation = dict(
            ax=order.open_time,
            ay=order.open,
            x=order.close_time if order.close else klines.iloc[-1]['date'],
            y=order.close if order.close else klines.iloc[-1]['open'],
            xref="x",
            yref="y",
            axref="x",
            ayref='y',
            text=text,
            showarrow=True,
            arrowhead=3,
            arrowwidth=1.5,
            arrowcolor=colors['sell'] if order.type == Actions.Sell else colors['buy'],
        )
        result.append(annotation)
    return result


def split_array(arr: np.ndarray, steps: int):
    n = len(arr)
    k = n // steps
    arrays = [arr[i*k:(i+1)*k] for i in range(steps)]
    return [(el.min(), el.max()) for el in arrays]


def _render_colored_candles(klines: pd.DataFrame, fig: go.Figure, prob_col: str) -> go.Figure:
    assert prob_col in klines.columns

    probability_range = np.sort(klines[prob_col].unique())

    steps = len(probability_range)
    prob_ranges = split_array(probability_range, steps)
    colors = interpolate_colors((252, 3, 3), (2, 237, 38), steps)
    for n in range(steps):
        df = klines[(klines[prob_col] >= prob_ranges[n][0]) &
                    (klines[prob_col] <= prob_ranges[n][1])]
        fig.add_traces(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing={
                'fillcolor': colors[n],
                'line': {
                    'color': colors[n],
                },
            },
            decreasing={
                'fillcolor': colors[n],
                'line': {
                    'color': colors[n],
                },
            }
        ))
    return fig


def render_indicators(df: pd.DataFrame, fig: go.Figure, indicators=[]) -> go.Figure:
    for ind in indicators:
        ind_line = go.Scatter(
            x=df['date'], y=df[ind['name']], mode='lines', line=dict(color=ind['color']))
        fig.add_trace(ind_line, row=1, col=1)

    return fig


def render_candles(
        df: pd.DataFrame,
        incrace_color: str = 'cyan',
        decrace_color: str = 'gray',
        orders: list[Order] | None = None,
        text_annotation: str = '',
        colored_deal_probability: bool = False,
        prob_col: str = '',
        indicators: list[dict] = [],


) -> go.Figure:
    """Render the candle chart

    Args:
        klines (pd.DataFrame): data for draw the chart
        incrace_color (str, optional): incrace candle color. Defaults to 'cyan'.
        decrace_color (str, optional): decrace candle color. Defaults to 'gray'.
        orders (list[Order] | None, optional): orders list. Defaults to None.
        text_annotation (bool, optional): Draw text annotation for deals.
        colored_deal_probability (bool, optional): draw colored candles for deal probability. Defaults to False.
        prob_col (str, optional): probability column in dataframe. Defaults to ''.
        indicators (list, optional): Indicators list. Contain list[dict], where each element is
         {'name': 'bb', 'color': 'yellow'} Defaults to [].

    Returns:
        go.Figure: chart figure
    """
    if colored_deal_probability:
        assert prob_col != ''

    # implement main candle chart
    _colors = {'inc': incrace_color, 'dec': decrace_color}
    main_chart = go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing={
            'fillcolor': _colors['inc'],
            'line': {
                'color': _colors['inc'],
            }
        },
        decreasing={
            'fillcolor': _colors['dec'],
            'line': {
                'color': _colors['dec'],
            }
        },
    )

    # check separately charts
    separately_indicators = []
    for indicator in indicators:
        if indicator.get('separately', False):
            separately_indicators.append(indicator['name'])

    # make subplots
    main_chart_places = 4
    add_specs = [*[[{}] if name != 'equity' else [{'rowspan': 2}]
                   for name in separately_indicators]]
    if 'equity' in separately_indicators:
        add_specs = [*add_specs, [None]]
    rows = main_chart_places + \
        len(separately_indicators) + \
        (1 if 'equity' in separately_indicators else 0)
    fig_general = make_subplots(
        rows=rows,
        cols=1,
        vertical_spacing=0.01,
        shared_xaxes=True,
        specs=[
            [{'rowspan': main_chart_places}],
            *[[None] for _ in range(main_chart_places - 1)],
            *add_specs,
        ],
        subplot_titles=[None, *[name for name in separately_indicators]]
    )

    # add main candle chart
    fig_general.add_trace(main_chart, row=1, col=1)

    # add another charts
    row = main_chart_places + 1
    for indicator in indicators:
        if indicator.get('separately', False):
            fig_general.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[indicator['name']],
                    mode='lines',
                    line=dict(color=indicator['color']),
                ),
                row=row,
                col=1,
            )
            row += 1
        else:
            ind_line = go.Scatter(
                x=df['date'], y=df[indicator['name']], mode='lines', line=dict(color=indicator['color']))
            fig_general.add_trace(ind_line, row=1, col=1)

    if colored_deal_probability:
        fig_general = _render_colored_candles(df, fig_general, prob_col)

    # set chart parameters
    fig_general.update_layout(
        xaxis_rangeslider_visible=False,
        annotations=order_annotations(
            df, orders, text_annotation) if orders else [],
        template="plotly_dark",
        hovermode='x unified',
    )
    fig_general.update_layout(
        {
            'plot_bgcolor': "#151822",
            'paper_bgcolor': "#151822",
        },
        font=dict(color='#dedddc'),
        showlegend=False,
        margin=dict(b=20, t=0, l=0, r=40),
    )
    fig_general.update_yaxes(
        showgrid=True,
        zeroline=False,
        showticklabels=True,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True,
        spikedash='dot',
        side='right',
        spikethickness=1,
    )

    return fig_general


class TesterBaseClass:
    """Backtest strategy for klines base class"""

    def __init__(
        self,
        klines: pd.DataFrame,
        start_kline: int = 0,
        depo: int = 1000,
        TP: float = 0.75,
        SL: float = 0.75,
        text_annotation: str = '',
        long_market_fee: float = 0.0004,
        short_market_fee: float = 0.0004,
        indicators: list = [],
    ) -> None:
        """
        Params:
            klines: pandas Dataframe contains the history candels
            start_kline: int - numer of star candale
            depo: int - start deposit
            TP: float - Take profit in percent of price
            SL: float - Stop loss in persent of price
            text_annotation: str - Write order text annotation ('pnl', 'amount', 'pnl+amount')
            indicators (dict['name': <collumn name>, 'color': <line color>]): Dict of indicators.
        """
        try:
            self._on_tick()
        except NotImplementedError:
            raise NotImplementedError
        except:
            pass

        def check_required_columns(colNames: list[str]) -> None:
            for colName in colNames:
                if colName not in klines:
                    raise Exception(
                        f'klines should contain a "{colName}" column.')

        if not isinstance(klines, pd.DataFrame):
            raise Exception('klines should be a padas Dataframe.')
        required_columns = ['date', 'open', 'close', 'high', 'low']
        check_required_columns(required_columns)

        self.klines = klines
        self.start_depo = depo
        self.balance = depo
        # tick number
        self.n_tick = start_kline
        self._tick = self.klines.iloc[self.n_tick]
        self.start_kline = start_kline
        self.orders = []
        self.next_id = 1
        self.TP = TP
        self.SL = SL
        self.margin_call = False
        self.equity: list[float] = [*[depo for _ in range(start_kline + 1)]]
        self.done = False
        self.do_render = False
        self._last_tick = len(klines) - 1
        self._text_annotation = text_annotation
        self._long_market_fee = long_market_fee
        self._short_market_fee = short_market_fee
        self.indicators = indicators

        for ind in self.indicators:
            if ind['name'] not in self.klines.columns:
                raise Exception(
                    f"'{ind['name']}' is in indicators but not in klines ")

    @property
    def tick(self) -> pd.Series:
        return self._tick

    def render(self) -> None:
        """Render the klines chart"""
        fig = render_candles(
            df=self.klines.assign(equity=self.equity),
            incrace_color='cyan',
            decrace_color='gray',
            orders=self.orders,
            text_annotation=self._text_annotation,
            indicators=[*self.indicators,
                        dict(name='equity', color='orange', separately=True)],
        )
        fig.show()
        self.print_info()

    def print_info(self) -> None:
        print(self.info(detail=1))

    def info(self, detail=2) -> dict:
        """return the trade statistic"""
        profit_orders = 0
        loss_orders = 0
        for order in self.orders:
            if order.close:
                if order.pnl > 0:
                    profit_orders += 1
                else:
                    loss_orders += 1

        return {
            'balance': round(self.balance, 8),
            'orders': len(self.orders) if detail < 2 else self.orders,
            'profit_orders': profit_orders,
            'loss_orders': loss_orders,
            'pnl': round(self.balance - self.start_depo, 8),
            'pnl_percent': round(self.balance * 100 / self.start_depo - 100, 2),
        }

    def get_open_orders(self) -> list[Order]:
        """Returns open order list"""
        result = []
        for order in self.orders:
            if not order.close:
                result.append(order)
        return result

    def _order_pnl(self, order: Order) -> tuple[float, float]:
        _fee = self._long_market_fee if order.type == Actions.Buy else self._short_market_fee
        _close = order.close if order.close else self._tick['open']
        fee_sum = order.vol * _fee + order.vol / order.open * _close * _fee
        pnl = order.vol / order.open * _close - order.vol - fee_sum
        pnl_percent = pnl / order.vol
        if order.type == Actions.Sell:
            pnl *= -1
            pnl_percent *= -1
        return pnl, pnl_percent

    def close_order(self, order: Order) -> float:
        """Close the order"""
        order.close = self._tick['open']
        order.close_time = self._tick['date']

        order.pnl, order.pnl_percent = self._order_pnl(order)
        self.balance += order.pnl

        return order.pnl

    def check_done(self) -> None:
        """Check and flip the Done flag"""
        if self.margin_call or self.n_tick >= self._last_tick:
            self.done = True

    def open_order(self, order_type: Actions, vol: float, TP: float = 0, SL: float = 0) -> Order:
        """Open order function"""
        # set TP and SL to inf if it's not set
        if TP == 0:
            if order_type == Actions.Buy:
                TP = float('inf')
        if SL == 0:
            if order_type == Actions.Sell:
                SL = float('inf')

        # make the order
        order = Order(
            id=self.next_id,
            type=order_type,
            open_time=self._tick['date'],
            open=self._tick['open'],
            vol=vol,
            TP=TP,
            SL=SL,
        )
        self.orders.append(order)
        self.next_id += 1
        return order

    def on_tick(self, *args, **kwargs) -> ...:
        if self.n_tick > self._last_tick:
            """ End of episode """
            raise Exception(
                f'n_tick error! n_tick={self.n_tick} but length of klines is {len(self.klines)}')

        self._tick = self.klines.iloc[self.n_tick]
        self.n_tick += 1

        # get open orders
        open_orders = self.get_open_orders()

        # Calculate P&L
        general_pnl = 0
        for order in open_orders:
            order.pnl, _ = self._order_pnl(order)
            general_pnl += order.pnl
        self.equity.append(self.balance + general_pnl)

        # margin call if depo <= 20%
        if self.balance <= self.start_depo * 0.2:
            self.margin_call = True

        result = self._on_tick(*args, **kwargs)

        # check Done
        self.check_done()

        if self.done and self.do_render:
            self.render()

        if self.margin_call:
            result = -max(self.equity)

        return result

    def _on_tick(self) -> ...:
        """
        Handle of the next tick
        Should return float reward (if use with Open AI Gym)
        """
        raise NotImplementedError


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
        Bolliger bands strategy. Open and close orders by bollinger bands indicator
    """

    def _on_tick(self, action: dict) -> float:
        """
        Handle of the next tick
        Args:
            action: dict:
                action: int (0-'Pass', 1-'Buy'),
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
                if bid >= tick['bb_upper'] or self.done:
                    tick_pnl += self.close_order(order)
            else:
                # Sell
                if bid <= tick['bb_lower'] or self.done:
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
