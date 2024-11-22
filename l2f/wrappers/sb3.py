import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info
import l2f
vector = l2f.vector1024


class L2F(VecEnv):

    actions: np.ndarray

    def __init__(self, seed=0):
        self.dtype = np.float32
        self.device = l2f.Device()
        self.rngs = vector.VectorRng()
        vector.initialize_rng(self.device, self.rngs, seed)
        self.envs = vector.VectorEnvironment()
        vector.initialize_environment(self.device, self.envs)
        self.parameters = vector.VectorParameters()
        self.states = vector.VectorState()
        self.next_states = vector.VectorState()
        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        vector.step(self.device, self.envs, self.parameters, self.states, self.actions, self.next_states, self.rngs)
        # for env_idx in range(self.num_envs):
        #     obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
        #         self.actions[env_idx]
        #     )
        #     # convert to SB3 VecEnv api
        #     self.buf_dones[env_idx] = terminated or truncated
        #     # See https://github.com/openai/gym/issues/3102
        #     # Gym 0.26 introduces a breaking change
        #     self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        #     if self.buf_dones[env_idx]:
        #         # save final observation where user can get it, then reset
        #         self.buf_infos[env_idx]["terminal_observation"] = obs
        #         obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        #     self._save_obs(env_idx, obs)
        observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        vector.observe(self.device, self.envs, self.parameters, self.next_states, observation, self.rngs)
        rewards = np.empty((self.envs.N_ENVIRONMENTS), dtype=self.dtype)
        vector.reward(self.device, self.envs, self.parameters, self.states, self.actions, self.next_states, rewards, self.rngs)
        dones = np.empty((self.envs.N_ENVIRONMENTS), dtype=bool)
        vector.terminated(self.device, self.envs, self.parameters, self.states, dones, self.rngs)
        buf_infos = [{} for _ in range(self.envs.N_ENVIRONMENTS)]
        for env_idx in range(self.envs.N_ENVIRONMENTS):
            if dones[env_idx]:
                buf_infos[env_idx]["terminal_observation"] = observation[env_idx].copy()
        vector.sample_initial_parameters_if_truncated(self.device, self.envs, self.parameters, dones, self.rngs)
        vector.sample_initial_state_if_truncated(self.device, self.envs, self.parameters, self.states, dones, self.rngs)

        next_observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        vector.observe(self.device, self.envs, self.parameters, self.next_states, next_observation, self.rngs)
        self.states.assign(self.next_states)
        return (next_observation, rewards, dones, buf_infos)




    def reset(self) -> VecEnvObs:
        vector.sample_initial_parameters(self.device, self.envs, self.parameters, self.rngs)
        vector.sample_initial_state(self.device, self.envs, self.parameters, self.states, self.rngs)
        observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        return observation


    def close(self):
        pass




    def get_images(self):
        return []


    def render(self, mode = None):
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [None for _ in self.envs.N_ENVIRONMENTS]
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        pass

    def env_is_wrapped(self, wrapper_class, indices = None):
        return [False for env_i in (range(self.envs.N_ENVIRONMENTS) if indices is None else indices)]


if __name__ == "__main__":
    env = L2F()
    env.reset()
    action = np.ones((env.envs.N_ENVIRONMENTS, env.envs.ACTION_DIM), dtype=env.dtype)
    for step in range(100):
        env.step(action)
    env.close()
    print("done")