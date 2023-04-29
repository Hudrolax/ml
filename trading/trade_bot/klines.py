import pandas as pd
import logging
from binance.um_futures import UMFutures
from binance.cm_futures import CMFutures
from binance.spot import Spot
import asyncio
from itertools import product
from .indicators import bollinger_bands, rsi, moving_average, average_true_range, macd, obv
from .preprocessing import make_observation_window
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

# adding func associations
indicator_func = {
    'bb': bollinger_bands,
    'rsi': rsi,
    'ma': moving_average,
    'atr': average_true_range,
    'macd': macd,
    'obv': obv,
}

registerd_symbols = {
    'DOGEUSDT': UMFutures,
    'BTCUSDT': UMFutures,
    'DOGEBTC': Spot
}

def calculate_indicators(klines: pd.DataFrame, kwargs: dict) -> tuple[pd.DataFrame, list]:
    """Calculate indicators for klines.

    Args:
        klines (pd.DataFrame): klines
        kwargs (dict):
            render (bool): render the indicator
            color (str): line color for rendering
            separately (bool): render independent chart below main chart
            **{other kwargs}: specify kwargs for indicator (like price, period, deviation, etc...)

    Returns:
        tuple[pd.DataFrame, list]: df - klines with indicator lines, list - list of indicators column names
        If column name contain `-p`, it's mean that column need scale in price scaler range.
    """
    kwargs = copy.deepcopy(kwargs)
    klines = klines.copy()
    indicators = []
    for key in kwargs.keys():
        value = kwargs[key]
        try:
            render = value.pop('render')
        except KeyError:
            render = False

        try:
            color = value.pop('color')
        except KeyError:
            color = 'white'

        try:
            separately = value.pop('separately')
        except KeyError:
            separately = False

        klines, cols = indicator_func[key](klines, **value)
        if render:
            for col in cols:
                indicators.append(
                    dict(name=col, color=color, separately=separately),)

    # if len(kwargs) > 0 and len(indicators) == 0:
    #     logger.warning(
    #         'Length of indicators list equals 0, but the data is preprocessed.')

    return klines, indicators

def download_klines(symbol: str, timeframe: str = '15m', limit: int = 500) -> pd.DataFrame | None:
    cols = ['open_time', 'open', 'high', 'low', 'close',
            'vol', 'close_time', 'qa_vol', 'trades',]
    clients = [CMFutures, UMFutures, Spot]

    logger.debug(f'Try to load history for {symbol}_{timeframe}')

    if symbol in registerd_symbols.keys():
        lib = registerd_symbols[symbol]
    else:
        # test client
        for client in clients:
            try:
                _client = client()
                res = _client.klines(symbol, timeframe, limit=1)
                lib = client
                logger.info(f'Use client: {client}')
                del _client
                break
            except Exception as ex:
                if 'Invalid symbol.' in ex.args:
                    continue
                else:
                    raise ex

    def columns_to_numeric(df: pd.DataFrame, col_names: list[str]):
        for col_name in col_names:
            df[col_name] = pd.to_numeric(df[col_name])
        return df

    try:
        client = lib()
        res = client.klines(symbol=symbol, interval=timeframe, limit=limit)
        df = pd.DataFrame(res).drop([9, 10, 11], axis=1)
        df.columns = cols
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df = df.rename({'open_time': 'date'}, axis=1)
        df = columns_to_numeric(df, ['open', 'high', 'low', 'close', 'vol', 'qa_vol', 'trades'])
        return df
    except Exception as e:
        logger.error(e)

async def download_klines_for_symbols_async(symbols: list[str], tfs: list[str], limit:int = 500) -> list[pd.DataFrame]:
    tasks = [asyncio.to_thread(download_klines, symbol, tf, limit) for symbol, tf in product(symbols, tfs)]
    results = await asyncio.gather(*tasks)
    return results

def download_klines_for_symbols(symbols: list[str], tfs: list[str], limit: int = 200) -> list[pd.DataFrame]:
    output = asyncio.run(download_klines_for_symbols_async(symbols, tfs, limit))
    return output

def download_and_preprocessing_klines(date: datetime, kwargs: dict) -> tuple[list[pd.DataFrame], pd.DataFrame] | tuple[None, None]:
    """Make and save observation dataframe for symbol / timeframe

    date (datetime): The open date of last closed candle
    Kwargs (dict):
        symbols (list): list of symbols for making a dataset. First symbol is general.
        tfs (list): list of timeframes for making a dataset. First timeframe is general.
        preprocessing_kwargs (dict): keyword arguments for making indicator Series

        example:
            kwargs = dict(
                symbols = ['DOGEUSDT', 'DOGEBTC', 'BTCUSDT'],
                tfs = ['15m', '30m', '1h', '4h'],
                preprocessing_kwargs = dict(
                                bb = dict(period=20, deviation=2),
                                rsi = dict(period=14),
                                ma = dict(period=20),
                                obv = dict(),
                            ),
            )
    """
    symbols = kwargs['symbols']
    tfs = kwargs['tfs']
    dfs = download_klines_for_symbols(symbols, tfs, 200)

    dfs2 = []
    for df in dfs:
        if df is None:
            return None, None

        df, _ = calculate_indicators(df, kwargs=kwargs['preprocessing_kwargs'])
        df = df.dropna()
        df = df[df['date'] <= date]
        dfs2.append(df)

    obs = make_observation_window(dfs2, date, 100)

    return dfs2, obs
