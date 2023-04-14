from trainer.data import load_history_for_symbols_list
import logging

kwargs = dict(
    symbols = ['BTCUSDT', 'DOGEBTC', 'DOGEUSDT'],
    tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
)
load_history_for_symbols_list(**kwargs)