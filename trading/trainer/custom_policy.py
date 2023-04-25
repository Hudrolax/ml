from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from .nn import *


registered_features_extractors = {
    'CustomCNN1d': CustomCNN1d,
    'CustomCNN2d': CustomCNN2d,
    'CustomCNN1dReverse': CustomCNN1dReverse,
    'Flatten': CustomFlatten,
}

registered_value_nets = {
    'mlp_64_64': mlp_64_64,
    'mlp_128_64': mlp_128_64,
    'mlp_256_64': mlp_256_64,
}


def get_custom_features_extractor(extractor_nn):
    """Function return a custom features extractor class

    Args:
        extractor_nn (torch NN class): Torch neural network class

    Returns:
        BaseFeaturesExtractor: Features extractor class
    """

    class CustomFeaturesExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            nn = extractor_nn(observation_space)
            super(CustomFeaturesExtractor, self).__init__(
                observation_space, nn.output_size)
            self.nn = nn

        def forward(self, observations):
            return self.nn(observations)

    return CustomFeaturesExtractor


def get_custom_mlp_extractor(mlp_nn):
    """Function returns a custom MLP extractor

    Args:
        mlp_nn (function): function returning a sequential layers of Extractor

    Returns:
        MlpExtractor: the MLP extractor class
    """
    class CustomMlpExtractor(MlpExtractor):
        def __init__(self, *args, **kwargs):
            super(CustomMlpExtractor, self).__init__(*args, **kwargs)

            # Change default net structure
            # self.shared_net = mlp_net(args[0])
            self.policy_net = mlp_nn(args[0])
            self.value_net = mlp_nn(args[0])

    return CustomMlpExtractor


def get_custom_policy(features_extractor: str, value_nn: str):
    """Function returns custom policy for RL model

    Args:
        features_extractor_class (str): name of registered features extractor class
        value_nn (str): name of function returning `value network`

    Returns:
        PolicyClass: Policy class for RL model
    """

    try:
        FeaturesExtractor = get_custom_features_extractor(
            registered_features_extractors[features_extractor])
    except KeyError:
        keys = [key for key in registered_features_extractors.keys()]
        raise KeyError(
            f'`{features_extractor}` not found among registered extractors. Try next: {keys}')

    try:
        value_net = registered_value_nets[value_nn]
    except KeyError:
        keys = [key for key in registered_value_nets.keys()]
        raise KeyError(f'`{value_nn}` not found among registered value nets. Try next: {keys}')
    MlpExtractor = get_custom_mlp_extractor(value_net)

    class CustomActorCriticPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            kwargs['features_extractor_class'] = FeaturesExtractor
            kwargs['normalize_images'] = False
            super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        def _build_mlp_extractor(self) -> None:
            self.mlp_extractor = MlpExtractor(self.features_dim, self.net_arch,
                                              self.activation_fn, self.device)

    return CustomActorCriticPolicy
