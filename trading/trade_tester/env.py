from .base_env import BaseTradingEnv
from gym.spaces import Discrete, Box
import numpy as np


class TradingEnv2DisceteActions(BaseTradingEnv):
    def set_action_space(self):
        self.action_space = Discrete(2)


class TradingEnv1BoxAction(BaseTradingEnv):
    def set_action_space(self):
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)


class TradingEnv2BoxAction(BaseTradingEnv):
    def set_action_space(self):
        self.action_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)