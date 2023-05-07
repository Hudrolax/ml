from trainer.pipeline import Pipeline
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)

pipeline_kwargs = dict(
    symbols=['DOGEUSDT'],
    tfs=['15m'],
    env_classes=['TradingEnv1BoxAction'],
    testers=['BBTesterMomentalReward'],
    features_extractors=['Flatten', 'CustomCNN1dReverse', 'CustomCNN2d', 'CustomCNN1d'],
    # features_extractors=['Flatten'],
    value_nets=['mlp_256_64'],
    gamma=0.95,
    gae_lambda=0.95,
    b_size=3000,
    batch_size=128,
    n_steps=128*20,
    total_timesteps=int(3e6),
    indicators=dict(
        bb=dict(price='close', period=20, deviation=1.6, render=False),
    ),
    continue_learning=True,
    # dataset_shape='100x6',
)

pipeline = Pipeline(**pipeline_kwargs)
pipeline.fit()
