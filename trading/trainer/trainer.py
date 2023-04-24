from trainer.env import make_env
from trainer.policy import get_model
from trainer.callbacks import SaveBestModelCallback


def train_model(total_timesteps: int, model_kwargs=dict(), env_kwargs=dict()):
    """Function getting a model and environment and train it."""
    env = make_env(**env_kwargs)
    model = get_model(env=env, **model_kwargs)

    save_path = model_kwargs.get('save_path', 'best_model/')
    save_name = model_kwargs.get('save_name', 'sac')
    callback = SaveBestModelCallback(save_path=save_path, save_name=save_name)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model