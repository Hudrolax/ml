import pandas as pd
import talib


def bollinger_bands(df, price='close', period=20, deviation=2):
    # Копирование исходного датафрейма
    df_copy = df.copy()
    
    # Расчет скользящего среднего
    df_copy['bb_middle'] = df_copy[price].rolling(window=period).mean()
    
    # Расчет стандартного отклонения
    df_copy['bb_std'] = df_copy[price].rolling(window=period).std()
    
    # Расчет верхней и нижней границ Bollinger Bands
    df_copy['bb_upper'] = df_copy['bb_middle'] + (df_copy['bb_std'] * deviation)
    df_copy['bb_lower'] = df_copy['bb_middle'] - (df_copy['bb_std'] * deviation)
    
    # Удаление вспомогательной колонки со стандартным отклонением
    df_copy.drop(columns=['bb_std'], inplace=True)

    return df_copy

def preprocessing(klines: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    klines = klines.copy()
    if 'bb' in kwargs.keys():
        bb = kwargs['bb']
        # klines = make_bb(klines, bb['price'], bb['period'], bb['deviation'])
        klines = bollinger_bands(klines, bb['price'], bb['period'], bb['deviation'])
    return klines


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
    klines = preprocessing(klines, kwargs=preprocessing_kwargs)
    klines = klines.iloc[-last_n:]

    # split data
    validate_len = int(len(klines) * split_validate_percent / 100)
    train_len = len(klines) - validate_len

    klines_train = klines.iloc[:train_len]
    klines_validate = klines.iloc[train_len:]

    # define indicators
    indicators = [
        dict(name='bb_upper', color='yellow'),
        dict(name='bb_middle', color='yellow'),
        dict(name='bb_lower', color='yellow'),
    ]

    return klines_train, klines_validate, indicators