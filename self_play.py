import os
import glob

import gymnasium as gym
from sb3_contrib import MaskablePPO


class SelfPlayWrapperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    use_trained_opponent = False
    latest_opponent_model = None
    latest_opponent_model_path = None
    opponent_model_path_pattern = None

    @staticmethod
    def update_opponent(randomize_policy=True):
        SelfPlayWrapperEnv.use_trained_opponent = not randomize_policy
        SelfPlayWrapperEnv.load_opponent_model()

    @staticmethod
    def load_opponent_model():
        SelfPlayWrapperEnv.latest_opponent_model = None
        SelfPlayWrapperEnv.latest_opponent_model_path = None
        if SelfPlayWrapperEnv.use_trained_opponent and SelfPlayWrapperEnv.opponent_model_path_pattern is not None:
            try:
                latest_policy = max(
                    glob.glob(f"{SelfPlayWrapperEnv.opponent_model_path_pattern}*.zip"), key=os.path.getctime
                )
                if latest_policy is not None:
                    SelfPlayWrapperEnv.latest_opponent_model = MaskablePPO.load(
                        latest_policy)
                    if SelfPlayWrapperEnv.latest_opponent_model is not None:
                        SelfPlayWrapperEnv.latest_opponent_model.set_random_seed(
                            42)
                        print('Updated opponent model to: {}'.format(latest_policy))
                        SelfPlayWrapperEnv.latest_opponent_model_path = latest_policy
                        print()
            except ValueError:
                pass
        else:
            print('Using random opponent')
            print()

    def __init__(self, env_create_func, primary_agent_index=0, use_trained_opponent=False, opponent_model_path_pattern=None, **kwargs):
        self._env = env_create_func(**kwargs)

        self.observation_space = self._env.observation_space(self._env.possible_agents[0])[
            "observation"
        ]
        self.action_space = self._env.action_space(
            self._env.possible_agents[0])

        self._primary_agent_index = primary_agent_index
        SelfPlayWrapperEnv.use_trained_opponent = use_trained_opponent
        SelfPlayWrapperEnv.opponent_model_path_pattern = opponent_model_path_pattern

        SelfPlayWrapperEnv.load_opponent_model()

    def _update_to_latest_best_opponent(self, randomize_policy=True):
        SelfPlayWrapperEnv.use_trained_opponent = not randomize_policy
        self._load_opponent_model()

    def step(self, action):
        current_cumulative_reward = self._env._cumulative_rewards[self._primary_agent]
        # print('{}: {}'.format(self._primary_agent, action))
        self._env.step(action)
        self._play_opponent_actions()

        observation, reward, done, truncated, info = self._env.last()

        info["is_success"] = reward == 1

        # The reward returned by the environment is the cumulative reward
        # since the last time the agent was reset.
        reward = reward - current_cumulative_reward

        return observation["observation"], reward, done, truncated, info

    def action_masks(self):
        return self._env.last()[0]["action_mask"]

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        self._primary_agent = self._env.agents[self._primary_agent_index]
        self._play_opponent_actions()
        observation, _, _, _, _ = self._env.last()
        return observation["observation"], {}

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _play_opponent_actions(self):
        while self._env.agent_selection != self._primary_agent:
            observation, _, termination, truncation, _ = self._env.last()
            if termination or truncation:
                action = None
            else:
                if SelfPlayWrapperEnv.latest_opponent_model is not None:
                    action = SelfPlayWrapperEnv.latest_opponent_model.predict(
                        observation["observation"],
                        action_masks=observation["action_mask"],
                    )[0].item()
                else:
                    mask = observation["action_mask"]
                    action = self._env.action_space(self._env.agent_selection).sample(
                        mask
                    )
            # time.sleep(1)
            # print('{}: {}'.format(self._env.agent_selection, action))
            self._env.step(action)
