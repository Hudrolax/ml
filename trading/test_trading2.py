import pandas as pd
import numpy as np
from trainer.env import make_env
from trainer.data import load_data
from trainer.trainer import train_model
from trainer.policy import get_model
import xarray as xr


dataset = xr.open_dataarray('dataset.nc')
load_data_kwargs = dict(
    path='klines/',
    symbol='DOGEUSDT',
    tf='15m',
    preprocessing_kwargs=dict(
        bb=dict(price='close', period=20, deviation=1.2, render=True),
    ),
    split_validate_percent=0,
    dataset=dataset,
)
train_klines, val_klines, indicators = load_data(**load_data_kwargs)

# Get only fitted for open/close klines
mask = ((train_klines['open'] >= train_klines['bb_upper']) |
        (train_klines['open'] <= train_klines['bb_lower']) |
        (train_klines['close'] >= train_klines['bb_upper']) |
        (train_klines['close'] <= train_klines['bb_lower']))
train_klines = train_klines[mask].reset_index(drop=True)

env_kwargs = dict(
    env_class='TradingEnv1BoxAction',
    tester='BBFutureTester3',
    klines=train_klines,
    data=dataset,
    expand_dims=True,
    indicators=indicators,
    b_size=100,
)

env = make_env(**env_kwargs)

# set model kwargs
model_kwargs = dict(
    load_model=True,
    gamma=0,
)

# train model
model = train_model(
    total_timesteps=int(6e6),
    env_kwargs=env_kwargs,
    model_kwargs=model_kwargs,
)

# model = get_model(env=env)

# done = False
# obs = env.reset()
# while not done:
#     # action = model.predict(obs)
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()

dataset.close()
