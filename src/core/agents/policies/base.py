import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BaseModel(BaseFeaturesExtractor):
    """ Base feature extractor (model) class for the multi-input betting environment. Each agent uses this class as
        the base model for processing the inputs.

        Attributes
        ----------
        :param observation_space: (Dict[str, np.ndarray] observation space of the environment.
                                  A dict of multiple inputs is expected.
        :param features_dim: (int) number of input features for the policy model
                             (the output of the final layer of this model).
        :param filters: (int) number of convolutional filters for the matrix convolutions.

    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int,
            filters: int
    ):
        super().__init__(observation_space=observation_space, features_dim=features_dim)

        # Extracting dimensions from the observation space.
        max_matches = observation_space.spaces['Fixture'].shape[1]
        fixture_channels = observation_space.spaces['Fixture'].shape[0]
        strategy_channels = observation_space.spaces['Strategy'].shape[0]

        # 1. Building Fixture Branch extractor: Conv1D + GELU + Flatten + Linear -> 128 Units.
        self.fixture_net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=fixture_channels, out_channels=filters, kernel_size=max_matches),
            torch.nn.GELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=filters, out_features=features_dim)
        )

        # 2. Building Strategy Branch extractor: Conv1D + GELU + Flatten + Linear -> 128 Units.
        self.strategy_net = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=strategy_channels, out_channels=filters, kernel_size=max_matches),
            torch.nn.GELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=filters, out_features=features_dim)
        )

        # 3. Building Strategy Branch extractor: Linear Embedding.
        self.step_net = torch.nn.Linear(in_features=max_matches, out_features=features_dim)

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        """ Passes each input to the corresponding branch and adds them together.

            Args
            ----
            :param observations: (Dict[str, np.ndarray]) dict of numpy observations coming from the gym environment.
        """

        fixture_out = self.fixture_net(observations['Fixture'])
        strategy_out = self.strategy_net(observations['Strategy'])
        step_out = self.step_net(observations['Step'])
        combined_features = fixture_out + strategy_out + step_out
        return combined_features
