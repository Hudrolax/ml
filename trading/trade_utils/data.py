import requests
import pandas as pd
import logging
from trainer.data import calculate_indicators


logger = logging.getLogger(__name__)
KLINES_ENDPOINT = 'http://hud.net.ru:8001/klines_csv/'


def get_raw_klines(
        symbol: str,
        tf: str,
        preprocessing_kwargs: dict = {},
) -> tuple[pd.DataFrame, list]:
    filepath = f'klines/{symbol}_{tf}.csv'
    try:
        logger.debug('Try to load the klines from cache.')
        df = pd.read_csv(filepath)
        logger.debug('klines loaded from cache.')
    except OSError:
        logger.debug(f"can't load klines from cahce for {symbol}_{tf}")
        # try to download from server
        logger.debug(f'try to download csv history from server.')
        try:
            response = requests.get(KLINES_ENDPOINT, params={
                                    'symbol': symbol, 'timeframe': tf})
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(
                    f'Download error. Status code is {response.status_code}.')
        except OSError as ex:
            logger.error(
                f'Error download {symbol}_{tf} from {KLINES_ENDPOINT}: {ex}')
            raise ex

        df = pd.read_csv(filepath)

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    df, indicators = calculate_indicators(
        klines=df, kwargs=preprocessing_kwargs)

    return df, indicators
