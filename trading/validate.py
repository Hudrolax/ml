from trainer.data import load_data
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

# validate model
validate_env_kwargs = dict(
    env_class='TradingEnv2Actions',
    klines=validate_klines,
    indicators=indicators,
    b_size=3000,
)

validate_model(
    env_kwargs=validate_env_kwargs,
    validate_times=1,
    render=True,
)