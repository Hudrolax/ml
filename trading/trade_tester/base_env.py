from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import importlib
import pandas as pd
import xarray as xr


class BaseTradingEnv(Env):

    def __init__(self, klines: pd.DataFrame, data: xr.DataArray | None, 
                 risk: float = 1, b_size: int | None=None, tester='GymFuturesTester', tester_kwargs=None,
                 verbose: bool = False) -> None:
        """Env for binance futures strategy tester
        Args:
            klines (pd.Dataframe): history klines and indicators
            data (xarray.dataraay | None): datset for making observations. Default None.
            risk (float): Order volume in percent of balance.
            b_size (int | None): klines batch size for tester. If None - all klines placed in tester.
            text_annotation (bool): draw text annottions
            tester (tester class | None): tester class. If None GymFuturesTester on default.
            verbose (int): level for printing additional information
        """
        super(BaseTradingEnv, self).__init__()

        self.set_action_space()

        self.verbose = verbose
        self.klines: pd.DataFrame = klines
        self.data: xr.DataArray | None = data
        self.state: np.ndarray

        tester_module = importlib.import_module('trade_tester.tester')
        self.tester_class = getattr(tester_module, tester) 

        self.tester = None
        self.tester_kwargs = tester_kwargs
        self._risk: float = risk
        self._b_size = b_size
        self.total_reward = 0
        self.reset()

        self.set_observation()
    
    def set_action_space(self):
        """ Define envirnoment action space
        Example:
            self.action_space = Discrete(3)
            or:
            self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        """
        raise NotImplementedError
    
    def set_observation(self):
        """Define environment observtion."""
        self.observation_space = Box(
            shape=self.state.shape,
            dtype=np.float32,
            low=0,
            high=1
        )

    def step(self, action: np.ndarray):
        """Apply action"""
        reward = self.tester.on_tick({
            'action': action,
            'risk': self._risk,
        })
        self.total_reward += reward

        """Set state"""
        self.state = self._get_observations()

        """Set info"""
        info = {}

        done = self.tester.done
        if self.verbose > 0 and done:
            self.tester.print_info()
            info = self.tester.info(detail=0)

        # Return step information
        return self.state, reward, done, info

    def render(self, mode='human') -> None:
        self.tester.do_render = True
    
    def _get_observations(self) -> np.ndarray:
        """Return an observation"""
        if self.data is not None:
            obs = self.data.sel(date=self.tester._tick['open_time']).values
        else:
            obs = np.array([1])
        return obs

    def reset(self) -> np.ndarray:
        klines = self.klines
        if self._b_size and self._b_size < len(self.klines):
            uncertain = len(klines) - self._b_size
            start = random.randint(0, uncertain)
            klines = klines.iloc[start: start + self._b_size]

        self.tester = self.tester_class(
            klines=klines,
            start_kline=0,
            **self.tester_kwargs
        )
        self.state = self._get_observations()
        self.total_reward = 0
        return self.state