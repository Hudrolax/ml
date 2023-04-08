from trainer.data import load_data
from trainer.trainer import train_model
from trainer.validate import validate_model
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)


# load data
load_data_kwargs = dict(
    path = 'klines/DOGEUSDT_15m.csv',
    preprocessing_kwargs = dict(
        bb = dict(price='close', period=20, deviation=2.0),
    ),
)
train_klines, validate_klines, indicators = load_data(**load_data_kwargs)

# set env kwargs
env_kwargs = dict(
    env_class='TradingEnv2Actions',
    klines=train_klines,
    indicators=indicators,
    window=1000,
    b_size=2000,
)

# set model kwargs
model_kwargs = dict(
    load_model=True,
)

# train model
model = train_model(
    total_timesteps=int(6e6),
    env_kwargs=env_kwargs,
    model_kwargs=model_kwargs,
)

# validate model
validate_env_kwargs = dict(
    env_class='TradingEnv2Actions',
    klines=validate_klines,
    indicators=indicators,
)

validate_model(
    model=model,
    env_kwargs=validate_env_kwargs,
    validate_times=10,
    render=False,
)

