from trainer.data import load_data
import pandas as pd
import numpy as np


indicators=dict(
    bb=dict(price='close', period=20, deviation=1.6, render=False),
    rsi=dict(price='close'),
    ma=dict(price='close'),
    obv=dict(),
    macd=dict(),
    # atr=dict(),
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
train_klines['bb_buy'] = 0
train_klines['bb_sell'] = 0
train_klines['next_5_max'] = pd.NA

i = 0
for index, row in train_klines.iterrows():
    # Buy
    if row['open'] <= row['bb_lower']:
        mask = (train_klines['date'] > row['date']) & (train_klines['open'] > train_klines['bb_middle'])
        close_kline = train_klines[mask]
        if not close_kline.empty and close_kline.iloc[0]['open'] > row['open']:
            train_klines.loc[index, 'bb_buy'] = 1

    #Sell
    elif row['open'] >= row['bb_lower']:
        mask = (train_klines['date'] > row['date']) & (train_klines['open'] < train_klines['bb_middle'])
        close_kline = train_klines[mask]
        if not close_kline.empty and close_kline.iloc[0]['open'] < row['open']:
            train_klines.loc[index, 'bb_sell'] = 1
    
    try:
        if train_klines.iloc[i + 5]['ma14'] > row['close']:
            train_klines.loc[index, 'next_5_max'] = 1
        else:
            train_klines.loc[index, 'next_5_max'] = 0
    except IndexError:
        pass
    i += 1

train_klines = train_klines.dropna()
train_klines.to_csv('bb_dataset.csv', index=False)