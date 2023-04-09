import pandas as pd
from .indicators import bollinger_bands, rsi


def preprocessing(klines: pd.DataFrame, kwargs: dict) -> tuple[pd.DataFrame, list]:
    klines = klines.copy()
    indicators = []
    keys = kwargs.keys()
    if 'bb' in keys:
        indicator = kwargs['bb']
        klines = bollinger_bands(klines, indicator['price'], indicator['period'], indicator['deviation'])
        indicators.append(dict(name='bb_upper', color='yellow'),)
        indicators.append(dict(name='bb_middle', color='yellow'),)
        indicators.append(dict(name='bb_lower', color='yellow'),)

    if 'rsi' in keys:
        indicator = kwargs['rsi']
        klines = rsi(klines, indicator['price'], indicator['period'])
        indicators.append(dict(name='rsi', color='white', separately=True),)


    return klines, indicators


def load_data(path: str, preprocessing_kwargs: dict = {}, split_validate_percent: int = 20, last_n: int | float=0):
    """Load, preprocessing and return train and validate dataframes and indicator dict

    Args:
        path (str): path fot loading klines
        preprocessing_kwargs (dict, optional): preprocessing kwargs. Defaults to {}.
        split_validate_percent (int, optional): Percent of validate df. Defaults to 20.
        last_n (int | float, optional): Use inly last_n klines. Defaults to 0.

    Returns:
        _type_: tuple (train_klines, val_klines, indicators_dict_for_env)
    """
    # load data
    last_n = int(last_n)
    klines = pd.read_csv(path)[['open_time', 'open', 'high', 'low', 'close', 'vol', 'trades']]
    klines = klines.rename({'open_time': 'date'}, axis=1)
    klines['date'] = pd.to_datetime(klines['date'], unit='ms')

    # meaking features
    klines, indicators = preprocessing(klines, kwargs=preprocessing_kwargs)
    klines = klines.iloc[-last_n:]

    # split data
    validate_len = int(len(klines) * split_validate_percent / 100)
    train_len = len(klines) - validate_len

    klines_train = klines.iloc[:train_len]
    klines_validate = klines.iloc[train_len:]


    return klines_train, klines_validate, indicators