import numpy as numpy
import pandas as pd
from .websocket import get_kline_ticker, on_new_kline
from .klines import download_and_preprocessing_klines
from time import sleep
from datetime import datetime


class TradeBotBaseClass:
    def __init__(self) -> None:
        self.klines: pd.DataFrame = pd.DataFrame([])
        self.symbol: str = ''
        self.tf: str = ''
        self.tick: dict = {}
        self.observation_kwargs: dict = {}
        self.dfs: list[pd.DataFrame] = []
        self.obs: pd.DataFrame = pd.DataFrame([])
    
    def set_observation_kwargs(self, **observation_kwargs) -> None:
        self.observation_kwargs = observation_kwargs
        self.symbol = observation_kwargs['symbols'][0]
        self.tf = observation_kwargs['tfs'][0]

    
    def start(self):
        assert len(self.observation_kwargs.keys()) != 0, 'Error: observation kwargs must be filled by the set_observation_kwargs method!'
        assert 'symbols' in self.observation_kwargs.keys(), 'Error: symbols must be into observation kwargs!'
        assert 'tfs' in self.observation_kwargs.keys(), 'Error: tfs must be into observation kwargs!'

        self.ticker = get_kline_ticker(symbol=self.symbol, tf=self.tf, callback=self._on_tick)
        while True:
            sleep(1)
    
    def _on_tick(self, new_kline: bool, open_time: datetime, close_time: datetime):
        self.tick['new_kline'] = new_kline
        self.tick['open_time'] = open_time
        self.tick['close_time'] = close_time
        if new_kline:
            print('new kline')
            dfs, obs = download_and_preprocessing_klines(open_time, self.observation_kwargs)
            if dfs is not None and obs is not None:
                self.dfs = dfs
                self.obs = obs
                self.klines = self.dfs[0]
            else:
                return
        self.on_tick()
    
    @on_new_kline
    def on_tick(self, *args, **kwargs):
        """The function is called when a new tick is coming from the broker."""
        print('tick function')
        print(self.tick['open_time'])
        print(self.klines.tail(3))
        print(self.obs)
        exit()
        # raise NotImplementedError

