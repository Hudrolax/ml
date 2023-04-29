from trainer.data import make_observation_dataset
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)

kwargs = dict(
    window = 100,
    symbols = ['DOGEUSDT'],
    tfs = ['15m'],
    preprocessing_kwargs = dict(
                    # bb = dict(period=20, deviation=2),
                    # rsi = dict(period=14),
                    # ma = dict(period=20),
                    # obv = dict(),
                ),
)

make_observation_dataset(**kwargs)