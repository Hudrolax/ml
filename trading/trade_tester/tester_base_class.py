import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_classes import Order, Actions
import numpy as np
from matplotlib.colors import ColorConverter
import math


class BaseTesterException(Exception):
    pass


def count_zeros(number) -> int:
    """Function counts zeros before significant digit"""
    if number >= 1 or number <= 0:
        return 0
    return math.ceil(abs(math.log10(number))) - 1


def sround(x: float) -> float:
    """Smart round."""
    try:
        return round(x, count_zeros(x) + 3)
    except ValueError:
        return 0


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
            x=order.close_time if order.close else klines.iloc[-1]['open_time'],
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
            x=df['open_time'],
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
            x=df['open_time'], y=df[ind['name']], mode='lines', line=dict(color=ind['color']))
        fig.add_trace(ind_line, row=1, col=1)

    return fig


def render_candles(
        df: pd.DataFrame,
        incrace_color: str = 'cyan',
        decrace_color: str = 'gray',
        orders: list[Order] | None = None,
        text_annotation: str = '',
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
        prob_col (str, optional): probability column in dataframe. Defaults to ''.
        indicators (list, optional): Indicators list. Contain list[dict], where each element is
         {'name': 'bb', 'color': 'yellow'} Defaults to [].

    Returns:
        go.Figure: chart figure
    """
    # implement main candle chart
    _colors = {'inc': incrace_color, 'dec': decrace_color}
    main_chart = go.Candlestick(
        x=df['open_time'],
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
    # add_specs = [*[[{}] if name != 'equity' else [{'rowspan': 2}]
    #                for name in separately_indicators]]
    add_specs = [*[[{}] for _ in separately_indicators]]
    # if 'equity' in separately_indicators:
    #     add_specs = [*add_specs, [None]]
    rows = main_chart_places + \
        len(separately_indicators)
    # + (1 if 'equity' in separately_indicators else 0)
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
        if indicator.get('prob_col', None) is not None:
            prob_col = indicator['prob_col']
        elif indicator.get('separately', False):
            fig_general.add_trace(
                go.Scatter(
                    x=df['open_time'],
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
                x=df['open_time'], y=df[indicator['name']], mode='lines', line=dict(color=indicator['color']))
            fig_general.add_trace(ind_line, row=1, col=1)

    if prob_col:
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
                    raise BaseTesterException(
                        f'klines should contain a "{colName}" column.')

        if not isinstance(klines, pd.DataFrame):
            raise BaseTesterException('klines should be a padas Dataframe.')
        required_columns = ['open_time', 'open', 'close', 'high', 'low']
        check_required_columns(required_columns)

        if len(klines) == 0:
            raise BaseTesterException('TESTER: klines length is 0!')

        self.klines = klines
        self.start_depo = depo
        self.balance = depo
        # tick number
        self.n_tick = start_kline
        self._tick = self.klines.iloc[self.n_tick]
        self.start_kline = start_kline
        self.orders = []
        self.open_orders = []
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
            if ind.get('prob_col', None) is None and ind['name'] not in self.klines.columns:
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

    def print_info(self) -> None:
        print(self.info(detail=1))

    def info(self, detail=2) -> dict:
        """return the trade statistic"""
        profit_orders = 0
        loss_orders = 0
        orders_pnl = []
        for order in self.orders:
            orders_pnl.append(order.pnl)
            if order.close:
                if order.pnl > 0:
                    profit_orders += 1
                else:
                    loss_orders += 1

        pnl_std = sround(np.array(orders_pnl).std())
        mean_pnl = sround(np.array(orders_pnl).mean())

        # Statistic for Sharpe and Srotrino ratios
        equity_arr = np.array(self.equity)
        log_returns = np.log(equity_arr[1:]) - np.log(equity_arr[:-1])
        expected_return = np.mean(log_returns)

        # Sharpe ratio
        sharpe = sround(expected_return / log_returns.std())

        # Sortino ratio
        downside_deviation = np.sqrt(np.mean(np.minimum(0, log_returns)**2))
        sortino = sround(expected_return / downside_deviation)

        pnl = sround(self.balance - self.start_depo)
        pnl_percent = sround(self.balance * 100 / self.start_depo - 100)
        try:
            pl_ratio = sround(profit_orders / (profit_orders + loss_orders))
        except ZeroDivisionError:
            pl_ratio = 0

        return {
            'balance': sround(self.balance),
            'orders': len(self.orders) if detail < 2 else self.orders,
            'profit_orders': profit_orders,
            'loss_orders': loss_orders,
            'pl_ratio': pl_ratio,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'ticks': self.n_tick - self.start_kline,
            'mean pnl': mean_pnl,
            'PNL_std': pnl_std,
            'sharp': sharpe,
            'sortino': sortino,
        }

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

    def close_order(self, order: Order, price: float | None = None) -> float:
        """Close the order"""
        order.close = self._tick['open'] if price is None else price
        order.close_time = self._tick['open_time']

        order.pnl, order.pnl_percent = self._order_pnl(order)
        self.balance += order.pnl
        self.orders.append(order)
        self.open_orders.remove(order)

        return order.pnl

    def check_done(self) -> None:
        """Check and flip the Done flag"""
        if self.margin_call or self.n_tick >= self._last_tick:
            self.done = True

    def open_order(
        self,
        order_type: Actions,
        vol: float,
        TP: float = 0,
        SL: float = 0,
        price: float | None = None,
    ) -> Order:
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
            open_time=self._tick['open_time'],
            open=self._tick['open'] if price is None else price,
            vol=vol,
            TP=TP,
            SL=SL,
        )
        self.open_orders.append(order)
        self.next_id += 1
        return order

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

        if self.margin_call:
            result = -100

        reward = self._on_tick(*args, **kwargs)

        if self.done and self.do_render:
            self.render()

        return reward

    def _on_tick(self) -> ...:
        """
        Handle of the next tick
        Should return float reward (if use with Open AI Gym)
        """
        raise NotImplementedError
