import pandas as pd
import numpy as np
from trainer.env import make_env
from trainer.data import load_data


results = []
# for period in range(20, 220, 10):
#     for deviation in range(12, 21):
for period in range(20, 40, 10):
    for deviation in range(12, 14):
        str_res = []
        dev = deviation / 10
        str_res.append(period)
        str_res.append(dev)

        load_data_kwargs = dict(
            path = 'klines/BTCUSDT_1m.csv',
            preprocessing_kwargs = dict(
                bb = dict(price='close', period=period, deviation=dev),
            ),
            split_validate_percent = 0,
            # last_n = 1e5,
        )
        train_klines, val_klines, indicators = load_data(**load_data_kwargs)

        env_kwargs = dict(
            env_class='TradingEnv2Actions',
            tester='BBFutureTester2',
            klines=train_klines,
            window=3,
            indicators=indicators,
            b_size=7000,
        )

        env = make_env(**env_kwargs)
        obs = env.reset()

        total_reward = []
        for k in range(100):
            done = False
            obs = env.reset()
            total_ep_reward = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step([action])
                total_ep_reward += reward
                # env.render()
            total_reward.append(total_ep_reward)
        mean_ep_reward = np.array(total_reward).mean()
        print(f'period={period}, dev={dev}, mean_ep_reward={mean_ep_reward}')
        str_res.append(mean_ep_reward)
        results.append(str_res)
    df = pd.DataFrame(results, columns=['period', 'dev', 'mean_ep_rew'])
    df.to_csv('bb_optimization.csv', index=False)
    print(df)