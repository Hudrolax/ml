from trade_bot.bot import TradeBotBaseClass


kwargs = dict(
    symbols = ['DOGEUSDT', 'DOGEBTC', 'BTCUSDT'],
    tfs = ['1m', '30m', '1h', '4h'],
    preprocessing_kwargs = dict(
                    bb = dict(period=20, deviation=2),
                    rsi = dict(period=14),
                    ma = dict(period=20),
                    obv = dict(),
                ),
)
bot = TradeBotBaseClass()
bot.set_observation_kwargs(**kwargs)
bot.start()