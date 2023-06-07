from trainer.env import make_env
from trade_utils.data import get_raw_klines
import pandas as pd
from itertools import product
import logging
import requests
from datetime import datetime
from trade_tester.tester_base_class import BaseTesterException

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)


logger = logging.getLogger(__name__)

SYMBOLS_PATH = 'http://hud.net.ru:8001/symbols'

def check_sortiono(symbol: str, tf: str, period: int, dev: float) -> float:
    
    logger.info(f'Star check sortino for {symbol}_{tf} period {period} dev {dev}')
    load_data_kwargs = dict(
        symbol=symbol,
        tf=tf,
        preprocessing_kwargs=dict(
            bb=dict(period=period, render=True, deviation=dev),
        ),
    )
    klines, indicators = get_raw_klines(**load_data_kwargs)

    actual_date = datetime.fromisoformat('2023-01-01')
    klines = klines[klines['open_time'] >= actual_date]

    env_kwargs = dict(
        env_class='TradingEnv2BoxAction',
        tester='BBFreeActionTester',
        klines=klines,
        data=None,
        indicators=indicators,
        verbose=1,
        # b_size=1000,
    )
    env = make_env(**env_kwargs)

    done = False
    obs = env.reset()
    k = 1
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        if k % 10000 == 0:
            logger.info(f'Step {k}:{len(klines)} ({int(k * 100 / len(klines))}%)')
        k += 1

    return info[0]['sortino']


def search_best_bb(
    symbols: list[str],
    tfs: list[str],
    per_start: int,
    per_end: int,
    per_step: int,
    dev_start: float,
    dev_end: float,
    dev_step: float,
    path: str = 'sortino_searc2.csv',
) -> None:
    cols = ['symbol', 'tf', 'period', 'dev', 'sortiono']
    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame([], columns=cols)
    periods = [period for period in range(
        per_start, per_end + per_step, per_step)]
    deviations = [dev for dev in range(
        int(dev_start*10), int(dev_end*10), int(dev_step*10))]

    for symbol, tf, period, dev in product(symbols, tfs, periods, deviations):
        mask = (df['symbol'] == symbol) & (df['tf'] == tf) & (
            df['period'] == period) & (df['dev'] == dev)
        if len(df[mask]) > 0:
            continue

        try:
            sortino = check_sortiono(symbol=symbol, tf=tf,
                                     period=period, dev=dev/10)
            new = pd.DataFrame([[symbol, tf, period, dev, sortino]], columns=cols)
            df = pd.concat([df, new])
            df.to_csv(path, index=False)
            print(f'Saved symbol {symbol}, tf {tf}, per={period}, dev={dev}')
        except BaseTesterException as ex:
            print(ex)
            continue


if __name__ == '__main__':

    symbols = requests.get(SYMBOLS_PATH).json()

    params = dict(
        symbols = symbols,
        tfs = ['15m'],
        per_start = 20,
        per_end = 50,
        per_step = 10,
        dev_start = 1.6,
        dev_end = 2.5,
        dev_step = 0.1,
    )
    search_best_bb(**params)
