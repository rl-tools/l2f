import warnings
import time

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import l2f
import l2f.ui_server
import json



class L2F(VecEnv):

    def __init__(self, n_envs, seed=0, render_port=8080, start_ui_server=True):
        self.dtype = np.float32
        self.device = l2f.Device()
        self.ui = l2f.UI()
        assert(n_envs in l2f.vector_selector, f"n_envs must be one of {list(l2f.vector_selector.keys())}")
        self.vector = l2f.vector_selector[n_envs]
        self.rngs = self.vector.VectorRng()
        self.vector.initialize_rng(self.device, self.rngs, seed)
        self.envs = self.vector.VectorEnvironment()
        self.vector.initialize_environment(self.device, self.envs)
        self.parameters = self.vector.VectorParameters()
        self.states = self.vector.VectorState()
        self.next_states = self.vector.VectorState()
        self.actions = None
        self.render_port = render_port
        self.ui_server = None
        self.ui_client = None
        self.ui_last_sync = None
        self.start_ui_server = start_ui_server
        self.episode_step = np.zeros((self.envs.N_ENVIRONMENTS,), dtype=np.int32)
        self.episode_step_limit = self.envs.EPISODE_STEP_LIMIT
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.envs.OBSERVATION_DIM,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(self.envs.ACTION_DIM,), dtype=np.float32)
        super().__init__(self.envs.N_ENVIRONMENTS, observation_space, action_space)

    def step_async(self, actions):
        assert(not np.isnan(actions).any())
        self.actions = actions

    def step_wait(self):
        self.vector.step(self.device, self.envs, self.parameters, self.states, self.actions, self.next_states, self.rngs)
        observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        self.vector.observe(self.device, self.envs, self.parameters, self.next_states, observation, self.rngs)
        assert(not np.isnan(observation).any())
        rewards = np.empty((self.envs.N_ENVIRONMENTS), dtype=self.dtype)
        self.vector.reward(self.device, self.envs, self.parameters, self.states, self.actions, self.next_states, rewards, self.rngs)
        dones = np.empty((self.envs.N_ENVIRONMENTS), dtype=bool)
        self.vector.terminated(self.device, self.envs, self.parameters, self.states, dones, self.rngs)
        buf_infos = [{} for _ in range(self.envs.N_ENVIRONMENTS)]
        for env_idx in range(self.envs.N_ENVIRONMENTS):
            if dones[env_idx]:
                buf_infos[env_idx]["terminal_observation"] = observation[env_idx]
                buf_infos[env_idx]["TimeLimit.truncated"] = False
            else:
                if self.episode_step[env_idx] >= self.episode_step_limit:
                    buf_infos[env_idx]["TimeLimit.truncated"] = True
                    buf_infos[env_idx]["terminal_observation"] = observation[env_idx]
                else:
                    buf_infos[env_idx]["TimeLimit.truncated"] = False

        dones = np.logical_or(dones, self.episode_step >= self.episode_step_limit)
        self.episode_step += 1
        self.episode_step *= (1-dones)

        self.vector.sample_initial_parameters_if_truncated(self.device, self.envs, self.parameters, dones, self.rngs)
        self.vector.sample_initial_state_if_truncated(self.device, self.envs, self.parameters, self.next_states, dones, self.rngs)

        next_observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        self.vector.observe(self.device, self.envs, self.parameters, self.next_states, next_observation, self.rngs)
        assert(not np.isnan(next_observation).any())
        self.states.assign(self.next_states)
        return (next_observation, rewards, dones, buf_infos)




    def reset(self):
        self.vector.sample_initial_parameters(self.device, self.envs, self.parameters, self.rngs)
        self.vector.sample_initial_state(self.device, self.envs, self.parameters, self.states, self.rngs)
        observation = np.empty((self.envs.N_ENVIRONMENTS, self.envs.OBSERVATION_DIM), dtype=self.dtype)
        self.vector.observe(self.device, self.envs, self.parameters, self.states, observation, self.rngs)
        return observation

    def close(self):
        pass

    def get_images(self):
        return []
    
    def _get_event_loop(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def render(self, mode = None):
        if mode == "human":
            from websocket import create_connection
            if self.start_ui_server and (self.ui_server is None or self.ui_server.is_alive() is False):
                if self.ui_server is not None:
                    warnings.warn("ui_server is not alive. Restarting...")
                print("Starting ui_server...")
                self.ui_server = l2f.ui_server.start_server_in_background(port = self.render_port, scenario = "")
            if self.ui_client is None:
                uri = f"ws://localhost:{self.render_port}/backend"
                max_retries = 5
                retry_delay = 0.1
                for attempt in range(max_retries):
                    try:
                        client = create_connection(uri)
                        print(f"Connected to {uri} on attempt {attempt + 1}.")
                        handshake = json.loads(client.recv())
                        namespace = handshake["data"]["namespace"]
                        self.ui.ns = namespace
                        ui_message = l2f.set_ui_message(self.device, self.envs.environments[0], self.ui)
                        client.send(ui_message)
                        self.ui_client = client
                        break
                    except ConnectionRefusedError:
                        print(f"Connection refused. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                return None
            else:
                now = time.time()
                if self.ui_last_sync is None or now - self.ui_last_sync > 1:
                    parameters_message = l2f.set_parameters_message(self.device, self.envs.environments[0], self.parameters.parameters[0], self.ui)
                    self.ui_client.send(parameters_message)
                    self.ui_last_sync = now
                state_action_message = l2f.set_state_action_message(self.device, self.envs.environments[0], self.parameters.parameters[0], self.ui, self.states.states[0], [0, 0, 0, 0])
                self.ui_client.send(state_action_message)
        return None

    def get_attr(self, attr_name, indices = None):
        return [None for _ in range(self.envs.N_ENVIRONMENTS)]
    def set_attr(self, attr_name, value, indices = None):
        pass

    def env_method(self, method_name, *method_args, indices = None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices = None):
        return [False for env_i in (range(self.envs.N_ENVIRONMENTS) if indices is None else indices)]


if __name__ == "__main__":
    import time
    env = L2F() #start_ui_server=False, render_port=13337)
    env.reset()
    action = np.ones((env.envs.N_ENVIRONMENTS, env.envs.ACTION_DIM), dtype=env.dtype)
    wait_time = 1
    print(f"Waiting for {wait_time} seconds...")
    time.sleep(wait_time)
    while True:
        env.reset()
        env.render(mode = "human")
        for step in range(100):
            env.render(mode="human")
            env.step(action)
            time.sleep(0.01)
    env.close()
    print("done")
