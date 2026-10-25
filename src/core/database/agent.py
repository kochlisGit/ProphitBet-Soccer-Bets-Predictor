import os
import json
import pickle
import shutil
import optuna
from typing import Any, Dict, Optional, Tuple
from src.core.models.dashboard import get_study_filepath, get_storage_url
from src.core.agents.agent import Agent, AgentConfig


class _AgentIndexUnpickler(pickle.Unpickler):
    """Load legacy configs without importing stored policy algorithms."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(('stable_baselines3.', 'sb3_contrib.')) and name in {'PPO', 'QRDQN'}:
            return f'{module}:{name}'
        return super().find_class(module, name)


class AgentDatabase:
    """ Agent database handler class.

        Functions:
            1. Loads all stored agents.
            2. Checks whether an agents exists by id.
            3. Stores a trained agents and its config.
            4. Loads a trained agents and its config.
            5. Update agents configs.
            7. Deletes specified agents by id.

        Attributes
        ----------
        :param league_id: (str) current league id.
    """

    def __init__(self, league_id: str):
        self._league_id = league_id

        with open('storage/configs/agents.json', mode='r', encoding='utf-8') as jsonfile:
            agents_cfg = json.load(jsonfile)

        # Initialize all available agents.
        self._agents_directory = agents_cfg['agents_directory']
        self._agents_index_filepath = agents_cfg['agents_index_filepath']

        # Create agent's directory file if it does not exist.
        if not os.path.exists(path=self._agents_directory):
            os.makedirs(name=self._agents_directory, exist_ok=True)

        # Initialize or Restore index.
        self._index = self._initialize_or_restore_index()

    @property
    def league_id(self) -> str:
        return self._league_id

    @property
    def index(self) -> Dict[str, Dict[str, Any]]:
        return self._index

    def _initialize_or_restore_index(self) -> Dict[str, Dict[str, Any]]:
        """ Initializes an empty index (Dict) or restores the previously stored index, """

        if not os.path.exists(path=self._agents_index_filepath):
            return {}
        else:
            with open(self._agents_index_filepath, 'rb') as pklfile:
                index = _AgentIndexUnpickler(pklfile).load()

            validated_index = {}
            for league_id in index:
                stored_agents = {
                    agent_id: agent_config
                    for agent_id, agent_config in index[league_id].items()
                    if os.path.exists(path=self._build_agent_directory(agent_id=agent_id, league_id=league_id))
                }
                if stored_agents:
                    validated_index[league_id] = stored_agents

            return validated_index

    def _build_agent_directory(self, agent_id: str, league_id: Optional[str] = None) -> str:
        """ Builds the directory of an agent using its league id and agent id (user-defined names). """

        if league_id is None:
            league_id = self._league_id

        return f'{self._agents_directory}/{league_id}/agents/{agent_id}'

    def _update_index(self):
        """ Updates the agent index and stores it in the pre-defined path. """

        with open(self._agents_index_filepath, 'wb') as pklfile:
            pickle.dump(self._index, pklfile)

    def get_agent_ids(self) -> list[str]:
        """ Gets a list of all agent ids, sorted in ascending order. """

        agent_ids = self._index.get(self._league_id, [])
        return [] if len(agent_ids) == 0 else list(agent_ids.keys())

    def agent_exists(self, agent_id: str) -> bool:
        """ Checks whether a specified agent exists by id. """

        return self._league_id in self._index and agent_id in self._index[self._league_id]

    def save_agent(self, agent_config: AgentConfig, agent: Agent):
        """ Saves an agent by calling its save method and updates index.

            Args
            ----
            :param agent_config: (AgentConfig) agent's persisted configuration and reports.
            :param agent: (Agent) trained agent to save.
        """

        # Saving agent.
        checkpoint_directory = self._build_agent_directory(agent_id=agent_config.agent_id)
        os.makedirs(name=checkpoint_directory, exist_ok=True)
        agent.save(checkpoint_directory=checkpoint_directory)

        # Updating index.
        if self._league_id not in self._index:
            self._index[self._league_id] = {}

        self._index[self._league_id][agent_config.agent_id] = agent_config
        self._update_index()

    def load_agent(self, agent_id: str) -> Tuple[Agent, AgentConfig]:
        """ Reconstructs and loads an agent together with its stored configuration. """

        # Loading agent config.
        agent_config = self._index[self._league_id][agent_id]

        # Reconstructing the agent and its underlying policy.
        config = agent_config.agent_config
        agent = Agent.from_config(agent_config=config)
        agent._policy = agent.build(env=config['env'])

        # Loading the trained agent.
        checkpoint_directory = self._build_agent_directory(agent_id=agent_id)
        agent.load(checkpoint_directory=checkpoint_directory, env=config['env'])
        return agent, agent_config

    def delete_agent(self, agent_id: str):
        """ Deletes an existing agent and its study using its id. """

        shutil.rmtree(self._build_agent_directory(agent_id=agent_id), ignore_errors=True)
        self._index[self._league_id].pop(agent_id)
        self._update_index()

        study_filepath = get_study_filepath(league_id=self._league_id, entity='agents')

        if os.path.exists(study_filepath):
            storage_url = get_storage_url(study_filepath=study_filepath)

            for study_name in optuna.get_all_study_names(storage=storage_url):
                if study_name == agent_id:
                    try:
                        optuna.delete_study(study_name=agent_id, storage=storage_url)
                    except KeyError as _:
                        return
                    finally:
                        break
