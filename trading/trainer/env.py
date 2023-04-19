# from trade_tester.env import TradingEnv2Actions
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import importlib 
import logging


logger = logging.getLogger(__name__)


def make_env(**make_env_kwargs) -> DummyVecEnv:
    """ Make and return the Environment.
    Params:
        env_class (str): Environment classname from trade_tester.env module.
        klines (pd.DataFrame): klines data for environment.
        data (xarray): dataset for observation
        b_size (int): Batch size for tester. Length of klines for transfering to the tester instance.
        tester (str): Tester class name.
        depo (int): Deposit amount for tester instance. Read tester docs.
        TP (float): Take profit in percent for the tester. Read tester docs.
        SL (float): Stop loss in percent fot the tester. Read tester docs.
        indicators (dict['name': <collumn name>, 'color': <line color>]): Dict of indicators. Read tester docs.
        expand_dims (bool): Expand observation dims for using Conv2d layers.
    """
    expected_params = ['env_class', 'klines', 'data', 'b_size', 'tester',
                       'depo', 'TP', 'SL', 'indicators', 'num_envs', 'risk', 'expand_dims']
    for key in make_env_kwargs:
        if key not in expected_params:
            raise KeyError(
                f'Parameter `{key}` not expected for building environment.')

    logger.debug(f'Making new envrironment with kwargs:\n{make_env_kwargs}')

    env_class = make_env_kwargs['env_class']
    klines = make_env_kwargs['klines']
    data = make_env_kwargs.get('data', None)
    b_size = make_env_kwargs.get('b_size', 1000)
    tester = make_env_kwargs.get('tester', 'BBFutureTester')
    depo = make_env_kwargs.get('depo', 1000)
    TP = make_env_kwargs.get('TP', 0)
    SL = make_env_kwargs.get('SL', 0)
    indicators = make_env_kwargs.get('indicators', [])
    num_envs = make_env_kwargs.get('num_envs', 1)
    risk = make_env_kwargs.get('risk', 0.2)
    expand_dims = make_env_kwargs.get('expand_dims', False)

    env_kwargs = dict(
        klines=klines,
        data=data,
        b_size=b_size,
        tester=tester,
        risk=risk,
        expand_dims=expand_dims,
        tester_kwargs=dict(
            depo=depo,
            TP=TP,
            SL=SL,
            indicators=indicators,
        ),
    )

    tester_module = importlib.import_module('trade_tester.env')
    env_class_obj = getattr(tester_module, env_class) 
    env = DummyVecEnv([lambda: Monitor(env_class_obj(**env_kwargs))
                      for i in range(num_envs)])
    return env
