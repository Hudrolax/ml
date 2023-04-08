from trainer.env import make_env
from trainer.policy import get_model
import logging


logger = logging.getLogger(__name__)

def validate_model(model=None, validate_times: int=1, render=False, env_kwargs={}) -> float:
    """Function validate model and return mean reward per episode.

    Args:
        model (_type_): trained model
        validate_times (int, optional): How much times validate the model. Defaults to 1.
        render (bool, optional): Make render with best model? Defaults to False.
        env_kwargs (dict, optional): kwargs for making environment. Defaults to {}.
    """
    logger.info(f'Validate model with:')
    env = make_env(**env_kwargs)
    if model is None:
        model = get_model(env=env, load_model=True)
    
    total_reward:int = 0
    for i in range(validate_times):
        i_reward:int = 0
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            i_reward += reward[0]
            total_reward += reward[0]
            if done:
                logger.info(f'Episode {i} reward: {i_reward}')
            if i == validate_times - 1 and render:
                env.render()

    mean_reward = round(total_reward / validate_times, 1)
    logger.info(f'Mean validate episode reward: {mean_reward}')

    return mean_reward
