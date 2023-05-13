from trainer.data import load_data


indicators=dict(
    bb=dict(price='close', period=20, deviation=1.6, render=False),
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

train_klines.to_csv('bb_dataset.csv', index=False)