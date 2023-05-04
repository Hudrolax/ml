from trainer.env import make_env
from trainer.data import load_data, download_history_from_symbols_list
import pandas as pd
from itertools import product
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)


logger = logging.getLogger(__name__)

def check_sortiono(symbol: str, tf: str, period: int, dev: float) -> float:
    
    logger.info(f'Star check sortino for {symbol}_{tf} period {period} dev {dev}')
    load_data_kwargs = dict(
        path='klines/',
        symbol=symbol,
        tf=tf,
        preprocessing_kwargs=dict(
            bb=dict(period=period, render=True, deviation=dev),
        ),
        split_validate_percent=0,
    )
    try:
        train_klines, val_klines, indicators, dataset = load_data(
            **load_data_kwargs)
    except FileNotFoundError:
        download_history_from_symbols_list(symbols=[symbol], tfs=[tf])
        train_klines, val_klines, indicators, dataset = load_data(
            **load_data_kwargs)

    klines = train_klines

    env_kwargs = dict(
        env_class='TradingEnv2BoxAction',
        tester='BBFreeActionTester',
        klines=klines,
        data=dataset,
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
    path: str = 'sortino_search.csv',
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

        sortino = check_sortiono(symbol=symbol, tf=tf,
                                 period=period, dev=dev/10)
        new = pd.DataFrame([[symbol, tf, period, dev, sortino]], columns=cols)
        df = pd.concat([df, new])
        df.to_csv(path, index=False)
        print(f'Saved symbol {symbol}, tf {tf}, per={period}, dev={dev}')


if __name__ == '__main__':
    from binance.um_futures import UMFutures

    um_futures_client = UMFutures()
    info = um_futures_client.exchange_info()

    symbols = [symbol['symbol'] for symbol in info['symbols']]

    params = dict(
        symbols = symbols,
        tfs = ['15m'],
        per_start = 20,
        per_end = 60,
        per_step = 10,
        dev_start = 1.2,
        dev_end = 2.1,
        dev_step = 0.1,
    )
    search_best_bb(**params)