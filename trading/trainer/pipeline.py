from .env import make_env
from .data import load_data
from .trainer import train_model

import pandas as pd
import numpy as np
import logging
from itertools import product

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, **kwargs) -> None:
        try:
            self.symbols = kwargs['symbols']
            self.tfs = kwargs['tfs']
            self.env_classes = kwargs['env_classes']
            self.testers = kwargs['testers']
            self.features_extractors = kwargs['features_extractors']
            self.value_nets = kwargs['value_nets']
            self.b_size = kwargs['b_size']
            self.total_timesteps = kwargs['total_timesteps']
            self.indicators = kwargs.get('indicators', {})
            self.continue_learning = kwargs.get('continue_learning', False)
            self.dataset_shape = kwargs.get('dataset_shape', '')
        except KeyError as e:
            key = e.args[0]
            logger.critical(f'Pipeline: Unexpected kwarg `{key}`')

        self.result_collumns = ['symbol', 'tf', 'dataset_shape', 'env', 'tester', 'extractor',
                                'value_net', 'timesteps', 'mean_ep_rew', 'mean_balance',
                                'mean_orders', 'mean_pl_ratio', 'mean_sharp', 'mean_sortino',
                                'mean_ep_rew_rnd', 'mean_balance_rnd',
                                'mean_orders_rnd', 'mean_pl_ratio_rnd', 'mean_sharp_rnd', 'rnd_mean_sortino']
        self._init_result()

    def _init_result(self):
        try:
            self.result = pd.read_csv('pipeline.csv')
        except FileNotFoundError as e:
            logger.warning(f'Not found exists pipeline. Start a new one.')
            self.result = pd.DataFrame([], columns=self.result_collumns)

    def _search_result(self, symbol, tf, dataset_shape, env, tester, extractor, value_net):
        mask = (
            (self.result['symbol'] == symbol) &
            (self.result['tf'] == tf ) &
            (self.result['dataset_shape'] == dataset_shape ) &
            (self.result['env'] == env) &
            (self.result['tester'] == tester) &
            (self.result['extractor'] == extractor) &
            (self.result['value_net'] == value_net)
        )
        return self.result[mask], mask

    def _prepare_data(self, symbol, tf, preprocessing_kwargs):
        load_data_kwargs = dict(
            path='klines/',
            symbol=symbol,
            tf=tf,
            preprocessing_kwargs=preprocessing_kwargs,
            split_validate_percent=20,
            load_dataset=True,
            dataset_shape=self.dataset_shape,
        )
        return load_data(**load_data_kwargs)

    def _make_env(self, env_class, tester, klines, dataset, indicators):
        # set env kwargs
        env_kwargs = dict(
            env_class=env_class,
            tester=tester,
            klines=klines,
            data=dataset,
            indicators=indicators,
            verbose=1,
        )

        return make_env(**env_kwargs)

    def validate_model(self, env, times: int, model=None, random=False):
        m_reward = []
        m_balance = []
        m_orders = []
        m_pl_ratio = []
        m_sharp = []
        m_sortino = []
        for i in range(times):
            done = False
            obs = env.reset()
            ep_reward = 0
            while not done:
                if random:
                    action = [env.action_space.sample()]
                else:
                    action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                ep_reward += reward

            m_reward.append(ep_reward)
            m_balance.append(info[0]['balance'])
            m_orders.append(info[0]['orders'])
            m_pl_ratio.append(info[0]['pl_ratio'])
            m_sharp.append(info[0]['sharp'])
            m_sortino.append(info[0]['sortino'])

        def mn(arr: list) -> float:
            return np.array(arr).mean()

        return mn(m_reward), mn(m_balance), mn(m_orders), mn(m_pl_ratio), mn(m_sharp), mn(m_sortino)

    def fit(self):
        iterator = product(self.symbols, self.tfs, self.env_classes, self.testers,
                            self.features_extractors, self.value_nets)
        for symbol, tf, env_class, tester, fe, value_net in iterator:
            logger.info(f'Fit {symbol}_{tf}, {env_class}, {tester}, {fe}, {value_net}')

            result_timesteps = self.total_timesteps
            exist_result, mask = self._search_result(symbol, tf, self.dataset_shape, env_class, tester, fe, value_net) 
            if len(exist_result) != 0:
                if self.continue_learning:
                    logger.info('Result already exists. Continue learning...')
                    result_timesteps += exist_result['timesteps'].max()

                    # delete exist line from results
                    self.result = self.result.loc[~mask]
                else:
                    logger.info('Result already exists. Skip...')
                    continue

            train_klines, val_klines, indicators, dataset = self._prepare_data(
                symbol, tf, self.indicators)

            # set env kwargs
            env_kwargs = dict(
                env_class=env_class,
                tester=tester,
                klines=train_klines,
                data=dataset,
                indicators=indicators,
                b_size=self.b_size,
            )

            _load_model = True if self.continue_learning else False
            postfix = '' if self.dataset_shape == '' else f'_{self.dataset_shape}'
            model_kwargs = dict(
                load_model=_load_model,
                features_extractor=fe,
                value_net=value_net,
                save_name=f'ppo_{fe}_{value_net}{postfix}'
            )

            # train model
            model = train_model(
                total_timesteps=int(self.total_timesteps),
                env_kwargs=env_kwargs,
                model_kwargs=model_kwargs,
            )
            
            del train_klines

            val_env = self._make_env(
                env_class, tester, val_klines, dataset, indicators)

            # validate
            rew, balance, orders, pl_ratio, sharp, sortino = self.validate_model(
                val_env, 1, model)
            
            val_res_line = [
                rew,
                balance,
                orders,
                pl_ratio,
                sharp,
                sortino,
            ]

            # random policy
            rew2, balance2, orders2, pl_ratio2, sharp2, sortino2 = self.validate_model(
                val_env, 1, random=True)
            
            rnd_res_line = [
                rew2,
                balance2,
                orders2,
                pl_ratio2,
                sharp2,
                sortino2,
            ]

            result_line = [
                symbol,
                tf,
                self.dataset_shape,
                env_class,
                tester,
                fe,
                value_net,
                result_timesteps,
                *val_res_line,
                *rnd_res_line
            ]
            res_line = pd.DataFrame([result_line], columns=self.result_collumns)
            self.result = pd.concat([self.result, res_line], ignore_index=True).reset_index(drop=True)

            # save results
            self.result.to_csv('pipeline.csv', index=False)
            logger.info('Results save to pipeline.csv')
