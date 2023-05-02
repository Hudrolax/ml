from trainer.env import make_env
from trainer.data import load_data, download_history_from_symbols_list 
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)


def check_sortiono(symbol:str, tf:str, period: int, dev: float) -> float:
    load_data_kwargs = dict(
        path = 'klines/',
        symbol = symbol,
        tf=tf,
        preprocessing_kwargs = dict(
            bb = dict(period=period, render=True, deviation=dev),
        ),
        split_validate_percent = 0,
    )
    try:
        train_klines, val_klines, indicators, dataset = load_data(**load_data_kwargs)
    except FileNotFoundError:
        download_history_from_symbols_list(symbols=[symbol], tfs=[tf])
        train_klines, val_klines, indicators, dataset = load_data(**load_data_kwargs)

    env_kwargs = dict(
        env_class='TradingEnv2BoxAction',
        tester='BBFreeActionTester',
        klines=train_klines,
        data=dataset,
        indicators=indicators,
        verbose=1,
        # b_size=1000,
    )
    env = make_env(**env_kwargs)

    done=False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])

    return info[0]['sortino']

cols = ['period', 'dev', 'sortiono']

symbol = 'DOGEUSDT'
tf = '15m'
path = f'temp_sortino_{symbol}_{tf}.csv'
try:
    df = pd.read_csv(path)
except:
    df = pd.DataFrame([], columns=cols)

for period in range(20, 100, 10):
    for dev in range (10, 25, 1):
        mask = (df['period'] == period) & (df['dev'] == dev)
        if len(df[mask]) > 0:
            continue

        sortino = check_sortiono(symbol=symbol, tf=tf, period=period, dev=dev/10)
        new = pd.DataFrame([[period, dev, sortino]], columns=cols)
        df = pd.concat([df, new])
        df.to_csv(path, index=False)
        print(f'Saved per={period}, dev={dev}')