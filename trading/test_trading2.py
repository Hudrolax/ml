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
        bb=dict(price='close', period=20, deviation=2, render=True),
    ),
    split_validate_percent=20,
    dataset=dataset,
)
train_klines, val_klines, indicators = load_data(**load_data_kwargs)

env_kwargs = dict(
    env_class='TradingEnv1BoxAction',
    tester='BBTester',
    klines=train_klines,
    data=dataset,
    expand_dims=False,
    indicators=indicators,
    b_size=3000,
)

env = make_env(**env_kwargs)

# set model kwargs
model_kwargs = dict(
    # load_model=True,
)

# train model
model = train_model(
    total_timesteps=int(6e6),
    env_kwargs=env_kwargs,
    model_kwargs=model_kwargs,
)

# model = get_model(load_model=True, env=env)

# done = False
# obs = env.reset()
# while not done:
#     # action = model.predict(obs)
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()

dataset.close()
