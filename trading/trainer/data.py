from binance.um_futures import UMFutures
from binance.cm_futures import CMFutures
from binance.spot import Spot
import pandas as pd
from .indicators import bollinger_bands, rsi, moving_average, average_true_range, macd, obv
import logging
from time import sleep
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

    if len(kwargs) > 0 and len(indicators) == 0:
        logger.warning(
            'The length of the indicators list equals 0, but the data is preprocessed.')

    return klines, indicators


def load_data(
    path: str,
    symbol: str,
    tf: str,
    preprocessing_kwargs: dict = {},
    split_validate_percent: int = 20,
    last_n: int | float = 0,
    rename_columns=False,
    min_date=None,
    max_date=None,
    dataset=None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load, preprocessing and return train and validate dataframes and indicator dict

    Args:
        path (str): path fot loading klines
        preprocessing_kwargs (dict, optional): preprocessing kwargs. Defaults to {}.
        split_validate_percent (int, optional): Percent of validate df. Defaults to 20.
        last_n (int | float, optional): Use only last_n klines. Defaults to 0.
        rename_columns (bool): rename columns. Add symbol and timeframe.
        min_date (datetime | None): Minimum date in klines. Doesn't metter, if dataset not is None.
        max_date (datetime | None): Maximum date in klines. Doesn't metter, if dataset not is None.
        dataset (xr.dataarray | None): xarray for sync dates

    Returns:
        _type_: tuple (train_klines, val_klines, indicators_dict_for_env)
    """
    full_path = f'{path}{symbol}_{tf}.csv'
    # load data
    last_n = int(last_n)
    klines = pd.read_csv(
        full_path)[['open_time', 'open', 'high', 'low', 'close', 'vol', 'trades']]
    klines = klines.rename({'open_time': 'date'}, axis=1)
    klines['date'] = pd.to_datetime(klines['date'], unit='ms')

    # meaking features
    klines, indicators = calculate_indicators(
        klines,
        kwargs=preprocessing_kwargs,
    )
    if not dataset is None:
        klines = klines[(klines['date'] >= dataset.date.values.min()) & (klines['date'] <= dataset.date.values.max())]
    else:
        if not min_date is None:
            klines = klines[klines['date'] >= min_date]

        if not max_date is None:
            klines = klines[klines['date'] <= max_date]

    klines = klines.iloc[-last_n:]

    # rename columns
    if rename_columns:
        new_cols = {}
        for col in klines.columns:
            new_cols[col] = f'{symbol}_{tf}-{col}'
        klines = klines.rename(new_cols, axis=1)

    # drop NaN after calculating indicators
    klines = klines.dropna()

    # split data
    validate_len = int(len(klines) * split_validate_percent / 100)
    train_len = len(klines) - validate_len

    klines_train = klines.iloc[:train_len]
    klines_validate = klines.iloc[train_len:]

    return klines_train, klines_validate, indicators


def load_data_from_list(symbols: str, tfs: str, preprocessing_kwargs={}) -> list[pd.DataFrame]:
    result = []
    for symbol in symbols:
        for tf in tfs:
            load_data_kwargs = dict(
                path='klines/',
                symbol=symbol,
                tf=tf,
                preprocessing_kwargs=preprocessing_kwargs,
                split_validate_percent=0,
                # rename_columns = True,
            )

            df, _, _ = load_data(**load_data_kwargs)
            result.append(df)

    return result


def download_klines(file: str, symbol: str, timeframe: str = '15m') -> None:
    cols = ['open_time', 'open', 'high', 'low', 'close',
            'vol', 'close_time', 'qa_vol', 'trades',]
    clients = [CMFutures, UMFutures, Spot]

    logger.info(f'Try to load history for {symbol}_{timeframe}')

    # test client
    for client in clients:
        try:
            _client = client()
            res = _client.klines(symbol, timeframe)
            logger.info(f'Use client: {client}')
            break
        except Exception as ex:
            if 'Invalid symbol.' in ex.args:
                continue
            else:
                raise ex

    try:
        # try load klines from file
        df = pd.read_csv(file)
        logger.info('History loaded from CSV.')
    except:
        # except load last klines and save to file
        logger.info('There is no CSV file. Loading a new history.')
        res = _client.klines(symbol, timeframe)
        df = pd.DataFrame(res).drop([9, 10, 11], axis=1)
        df.columns = cols
        df.to_csv(file, index=False)
        df = pd.read_csv(file)

    # load older klines
    try:
        for i in range(99999999):
            min_timestamp = df['open_time'].min()-1
            res = _client.klines(symbol, timeframe, endTime=min_timestamp)
            if len(res) == 0:
                logger.info('All old klines is loaded.')
                break
            _df = pd.DataFrame(res).drop([9, 10, 11], axis=1)
            _df.columns = cols
            df = pd.concat([_df, df], ignore_index=True)
            df.to_csv(file, index=False)

            df_temp = df.head(1).copy()
            df_temp['open_time'] = pd.to_datetime(
                df_temp['open_time'], unit='ms')
            min_date = df_temp['open_time'].min()
            logger.info(f'Load old klines until {min_date}.')
            sleep(0.5)
    except Exception as ex:
        logger.error(ex)

    # load new klines
    try:
        for i in range(9999999):
            max_timestamp = df['open_time'].max() + 1
            res = _client.klines(symbol, timeframe, startTime=max_timestamp)
            if len(res) == 0:
                logger.info(f'All new klines loaded.')
                break
            _df = pd.DataFrame(res).drop([9, 10, 11], axis=1)
            _df.columns = cols
            df = pd.concat([df, _df], ignore_index=True)
            df.to_csv(file, index=False)
            df_temp = df.tail(1).copy()
            df_temp['open_time'] = pd.to_datetime(
                df_temp['open_time'], unit='ms')
            max_date = df_temp['open_time'].max()

            logger.info(f'Load new klines until {max_date}.')
            sleep(0.5)
    except Exception as ex:
        logger.error(ex)


def download_history_from_symbols_list(symbols: list[str], tfs: list[str], path='klines/') -> None:
    """Loding history for list of pairs/timeframes

    Args:
        pairs (list[str]): List of pairs for loading
        tfs (list[str]): list of timeframes for loading
    """
    for symbol in symbols:
        for tf in tfs:
            download_klines(
                file=path + f'{symbol}_{tf}.csv', symbol=symbol, timeframe=tf)
