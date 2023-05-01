from stable_baselines3 import PPO
from .custom_policy import get_custom_policy
import logging

logger = logging.getLogger(__name__)

def get_model(**model_kwargs):
    """Function load or define a new model.

    params:
        env: Environment for training or validate model
        load_model (bool): Load model or define a new one.
        save_path (str): Save path for saving best model.
        save_name (str): Name for saving the model.
        lr (int): Learning rate for learning the model.
        batch_size (int): Batch size for model learning. Read model.learn() help.
        n_steps (int): Count of steps for updating model weights. Reed model.learn() help.
        verbose (int): Verbose level for learning the model. Read model.learn() help.
        gamma (int): Gamma param for learning the model. Read model.learn() help.
        tensorboard_log (str): Path for saving tensorboard log. Read model.learn() help.
        n_epochs (str): Number of agent training epochs
        features_extractor (str): name of registered features extractor
        value_net (str): name of registered value neural network

    Returns:
        Model: Trained model.
    """
    expected_params = ['env', 'load_model', 'save_path', 'save_name', 'lr',
                        'batch_size', 'n_steps', 'verbose', 'gamma', 'tensorboard_log', 'n_epochs',
                        'features_extractor', 'value_net']
    for key in model_kwargs:
        if key not in expected_params:
            raise KeyError(f'Parameter `{key}` not expected for building model.')

    env = model_kwargs['env']
    load_model = model_kwargs.get('load_model', True)
    save_path = model_kwargs.get('save_path', 'best_model/')
    save_name = model_kwargs.get('save_name', 'ppo')
    lr = model_kwargs.get('lr', 3e-4)
    batch_size = model_kwargs.get('batch_size', 64)
    n_steps = model_kwargs.get('n_steps', batch_size * 100)
    n_epochs = model_kwargs.get('n_epochs', 2)
    features_extractor = model_kwargs.get('features_extractor', 'Flatten')
    value_net = model_kwargs.get('value_net', 'mlp_128_64')
    verbose = model_kwargs.get('verbose', 1)
    gamma = model_kwargs.get('gamma', 0.99)
    tensorboard_log = model_kwargs.get('tensorboard_log', 'tblog')

    path = save_path + save_name
    try:
        if load_model:
            message = f'Loding model from `{path}`...'
            logger.info(f'Loding model from `{path}`...')
            if logger.getEffectiveLevel() >= logging.WARNING:
                print(message)
            try:
                model = PPO.load(path)
                model.set_env(env)
            except:
                print('Error loading a model.')
                raise Exception('Create new model')
        else:
            raise Exception('Create new model')
    except:
        # Define a model
        message = f'Define a new model with kwargs:\n{model_kwargs}'
        logger.info(message)
        if logger.getEffectiveLevel() >= logging.WARNING:
            print(message)

        model = PPO(
            policy=get_custom_policy(features_extractor=features_extractor, value_nn=value_net),
            env=env,
            tensorboard_log=tensorboard_log,
            gamma=gamma,
            verbose=verbose,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=lr,
            n_epochs=n_epochs,
        )

    return model
