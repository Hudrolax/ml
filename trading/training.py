from trainer.pipeline import Pipeline
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                    datefmt='%d/%m/%y %H:%M:%S', level=logging.INFO)

pipeline_kwargs = dict(
    symbols=['DOGEUSDT'],
    tfs=['15m'],
    env_classes=['TradingEnv2BoxAction'],
    testers=['BBTester'],
    features_extractors=['Flatten', 'CustomCNN1dReverse', 'CustomCNN2d', 'CustomCNN1d'],
    value_nets=['mlp_64_64', 'mlp_128_64', 'mlp_256_64'],
    b_size=3000,
    total_timesteps=int(1e3),
    indicators=dict(
        bb=dict(price='close', period=20, deviation=2, render=False),
    ),
    continue_learning=True,
)

pipeline = Pipeline(**pipeline_kwargs)
pipeline.fit()