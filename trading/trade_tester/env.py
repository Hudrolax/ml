from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import importlib
import pandas as pd
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import math


# def plot_dataframe(df, draw=True):
#     """Return an observation as ndarray with image type and show image

#     Args:
#         df (pd.Dataframe): Dataframe with observation

#     Returns:
#         ndarray: 2D ndarray with observation img
#     """
#     fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
#     add_part = len(df.columns)-1
#     for col in df.columns:
#         data = df[col].copy()
#         data = data + add_part 
#         add_part -= 1
#         ax.plot(data, color='white', linewidth=1)
#     ax.set_facecolor('black')
#     ax.set_xlim([0, len(df)-1])
#     ax.set_ylim([0, len(df.columns)])
#     ax.axis('off')

#     # Получаем байтовую строку изображения и создаем объект Image
#     canvas = fig.canvas
#     fig.set_size_inches(512/80, 512/80) # 80 пикселей = 1 дюйм
#     canvas.draw()
#     w, h = canvas.get_width_height()
#     img_string = canvas.tostring_rgb()
#     img_array = np.frombuffer(img_string, dtype=np.uint8)
#     size = int(math.sqrt(img_array.shape[0]/3))
#     img_array = img_array.reshape((size, size, 3)) 
#     if draw:
#         plt.imshow(img_array, cmap='gray')
#         plt.show()

#     return img_array

# def scale_candles(scaler_range: tuple, candles: np.ndarray) -> np.ndarray:
#     def _transform_generator(arr):
#         for col in range(arr.shape[1]):
#             yield scaler.transform(arr[:, col].reshape(-1, 1))
#     flatten_arr = candles.flatten().reshape(-1, 1)
#     scaler = MinMaxScaler(scaler_range)
#     scaler.fit(flatten_arr)
#     return np.concatenate([col for col in _transform_generator(candles)], axis=1)

# def scale_columns_std(df, cols):
#     data = df[cols].values.astype(float)
#     scaler = StandardScaler()
#     scaler.fit(data.reshape(-1, 1))
#     data_scaled = scaler.transform(data.reshape(-1, 1)).reshape(-1, len(cols))
#     df_scaled = pd.DataFrame(data_scaled, index=df.index, columns=cols)
#     df_remaining = df.drop(cols, axis=1)
#     return pd.concat([df_remaining, df_scaled], axis=1)

# def normalize_data(df: pd.DataFrame):
#     """Normalize data. Candles one pair are normalizing in the same scale. Others data normalize independence."""
#     candle_fields = ['open', 'high', 'low', 'close']
#     candles = df[candle_fields].to_numpy()
#     scaler_range = (0.0001, 0.9999)
#     candles = scale_candles(scaler_range=scaler_range, candles=candles)

#     if 'date' in df.columns:
#         other_features = df.drop([*candle_fields, 'date'], axis=1).columns
#     else:
#         other_features = df.drop(candle_fields, axis=1).columns

#     for feature in other_features:
#         scaler = MinMaxScaler(scaler_range)
#         df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    
#     if 'date' in df.columns:
#         return pd.concat(
#             [
#                 df['date'].reset_index(drop=True),
#                 pd.DataFrame(candles, columns=candle_fields),
#                 df.drop('date', axis=1)[other_features].reset_index(drop=True),
#             ],
#             axis=1
#         )
#     else:
#         return pd.concat(
#             [
#                 pd.DataFrame(candles, columns=candle_fields),
#                 df[other_features].reset_index(drop=True),
#             ],
#             axis=1
#         )

# def make_features(df, window, cols) -> pd.DataFrame:
#     # Создаем пустой датафрейм, чтобы заполнить его новыми столбцами
#     new_df = pd.DataFrame()

#     # Добавляем новые столбцы, каждый из которых представляет собой конкатенацию
#     # предыдущих строк из соответствующей колонки
#     for col in cols:
#         col_data = [df[col].shift(i).fillna(np.nan)
#                     for i in range(1, window+1)]
#         new_col = pd.concat(col_data, axis=1, ignore_index=True)
#         new_col.columns = [f"{col}_prev_{i}" for i in range(1, window+1)]
#         new_df = pd.concat([new_df, new_col], axis=1)

#     # Объединяем новые столбцы с исходным датафреймом
#     result = pd.concat([df, new_df], axis=1)

#     return result


# def create_observation_window(df:pd.DataFrame, n:int, window_size:int) -> pd.DataFrame:
#     """функция для создания датасета на основе скользящего окна
#     Args:
#         df (pd.DataFrame): dataframe
#         n (int): from n string
#         window_size (int): window size

#     Returns:
#         ndarray: observation array size (len(df.columns), window_size)
#     """
#     return df.iloc[n-window_size:n, :].reset_index(drop=True)


class TradingEnv(Env):

    def __init__(self, klines: pd.DataFrame, data: xr.DataArray, 
                 risk: float = 1, b_size=None, tester='GymFuturesTester', tester_kwargs=None,
                 expand_dims=False) -> None:
        """Env for binance futures strategy tester
        Args:
            klines (pd.Dataframe): history klines and indicators
            data (xarray.dataraay): datset for making observations
            risk (float): Order volume in percent of balance.
            b_size (int | None): klines batch size for tester. If None - all klines placed in tester.
            text_annotation (bool): draw text annottions
            tester (tester class | None): tester class. If None GymFuturesTester on default.
            expand_dims (bool): Expand observation dims for using Conv2d layers.
        """
        super(TradingEnv, self).__init__()

        # 0 - Sell, 1 - Buy, 2 - Pass
        self.action_space = Discrete(3)
        self.klines: pd.DataFrame = klines
        self.data: xr.DataArray = data
        self.state: np.ndarray
        tester_module = importlib.import_module('trade_tester.tester')
        self.tester_class = getattr(tester_module, tester) 
        self.tester = None
        self.tester_kwargs = tester_kwargs
        self.expand_dims = expand_dims
        self._risk: float = risk
        self._b_size = b_size
        self.total_reward = 0
        self.reset()
        self.observation_space = Box(
            shape=self.state.shape,
            dtype=np.float32,
            low=0,
            high=1
        )

    def step(self, action: int | float):
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
        # info = self.tester.info(detail=0)

        done = self.tester.done

        # Return step information
        return self.state, reward, done, info

    def render(self, mode='human') -> None:
        self.tester.do_render = True
    
    # def plot_observation(self, draw=True):
    #     return plot_dataframe(self._get_observations(as_df=True), draw)

    def _get_observations(self) -> np.ndarray:
        """Return an observation"""
        obs = self.data.sel(date=self.tester._tick['date']).values
        if self.expand_dims:
            obs = np.expand_dims(obs, axis=0)
        return obs

    def reset(self):
        klines = self.klines
        if self._b_size and self._b_size < len(self.klines):
            uncertain = len(klines) - self._b_size
            start = random.randint(0, uncertain)
            klines = klines.iloc[start: start + self._b_size]

        self.tester = self.tester_class(
            klines=klines,
            start_kline=1,
            **self.tester_kwargs
        )
        self.state = self._get_observations()
        self.total_reward = 0
        return self.state


class TradingEnv2Actions(TradingEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 0 - Sell, 1 - Buy
        self.action_space = Discrete(2)


class TradingEnv1BoxAction(TradingEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 0 - Sell, 1 - Buy
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=float)