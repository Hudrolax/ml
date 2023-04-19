from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy 
from stable_baselines3.common.torch_layers import MlpExtractor
from .nn import CustomCNN2d, mlp_net
import logging


logger = logging.getLogger(__name__)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        cnn = CustomCNN2d(observation_space)
        super(CustomFeaturesExtractor, self).__init__(observation_space, cnn.output_size)
        self.cnn = cnn

    def forward(self, observations):
        return self.cnn(observations)


class CustomMlpExtractor(MlpExtractor):
    def __init__(self, *args, **kwargs):
        super(CustomMlpExtractor, self).__init__(*args, **kwargs)

        # Change default net structure
        # self.shared_net = mlp_net(args[0])
        self.policy_net = mlp_net(args[0])
        self.value_net = mlp_net(args[0])


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = CustomFeaturesExtractor
        kwargs['normalize_images'] = False
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(self.features_dim, self.net_arch,
                                                 self.activation_fn, self.device)


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

    Returns:
        Model: Trained model.
    """
    expected_params = ['env', 'load_model', 'save_path', 'save_name', 'lr',
                        'batch_size', 'n_steps', 'verbose', 'gamma', 'tensorboard_log']
    for key in model_kwargs:
        if key not in expected_params:
            raise KeyError(f'Parameter `{key}` not expected for building model.')

    env = model_kwargs['env']
    load_model = model_kwargs.get('load_model', True)
    save_path = model_kwargs.get('save_path', 'best_model/')
    save_name = model_kwargs.get('save_name', 'ppo')
    lr = model_kwargs.get('lr', 3e-4)
    batch_size = model_kwargs.get('batch_size', 32)
    n_steps = model_kwargs.get('n_steps', 4096)
    verbose = model_kwargs.get('verbose', 1)
    gamma = model_kwargs.get('gamma', 0.8)
    tensorboard_log = model_kwargs.get('tensorboard_log', 'tblog')

    try:
        if load_model:
            path = save_path + save_name
            model = PPO.load(path)
            model.set_env(env)
            logger.info(f'Loding model from `{path}`...')
        else:
            raise Exception('Create new model')
    except:
        # Define the model
        logger.info(f'Define new model with kwargs:\n{model_kwargs}')
        model = PPO(
            policy=CustomActorCriticPolicy,
            env=env,
            tensorboard_log=tensorboard_log,
            gamma=gamma,
            verbose=verbose,
            batch_size=batch_size,
            n_steps=n_steps,
            learning_rate=lr
        )

    return model
