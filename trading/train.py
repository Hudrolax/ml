from trainer.pipeline import Pipeline
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)

pipeline_kwargs = dict(
    symbols=['DOGEUSDT'],
    tfs=['15m'],
    env_classes=['TradingEnv1BoxAction'],
    testers=['BBTester'],
    # features_extractors=['Flatten', 'CustomCNN1dReverse', 'CustomCNN2d', 'CustomCNN1d'],
    features_extractors=['CustomCNN2d'],
    value_nets=['mlp_256_64'],
    b_size=1000,
    batch_size=128,
    total_timesteps=int(6e6),
    indicators=dict(
        bb=dict(price='close', period=20, deviation=1.6, render=False),
    ),
    continue_learning=True,
    # dataset_shape='100x6',
)

pipeline = Pipeline(**pipeline_kwargs)
pipeline.fit()
