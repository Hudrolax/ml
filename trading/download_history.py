from trainer.data import download_history_from_symbols_list
import logging

kwargs = dict(
    symbols = ['DOGEUSDT'],
    tfs = ['15m']
)
download_history_from_symbols_list(**kwargs)
