import pandas as pd
import xarray as xr
from trainer.data import load_data
from trainer.env import make_env

indicators = dict(
    bb=dict(price='close', period=20, deviation=1.6, render=True),
)

load_data_kwargs = dict(
    path='klines/',
    symbol='DOGEUSDT',
    tf='15m',
    preprocessing_kwargs=indicators,
    split_validate_percent=0,
    load_dataset=True,
)

train_klines, val_klines, indicators, dataset = load_data(**load_data_kwargs)

df = pd.read_csv('bb_dataset.csv')
dataset_path = f'data/DOGEUSDT_15m.nc'
dataset = xr.open_dataarray(dataset_path)

env_kwargs = dict(
    env_class='TradingEnv1BoxAction',
    tester='BBTester',
    klines=df,
    data=dataset,
    indicators=indicators,
    b_size=3000,
    verbose=1,
)
env = make_env(**env_kwargs)

done = False
obs = env.reset()
while not done:
    tick = env.envs[0].tester.klines.iloc[env.envs[0].tester.n_tick+0]
    action = [[1 if tick['bb_buy'] else 0.5]]
    obs, reward, done, info = env.step(action)
    env.render()

