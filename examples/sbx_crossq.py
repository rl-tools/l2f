from l2f.wrappers.sb3 import L2F
# from stable_baselines3 import SAC
from sbx import CrossQ
# from sb3_contrib import CrossQ
from stable_baselines3.common.vec_env import VecMonitor
import time
import torch



n_envs = 16
vec_env = L2F(n_envs)
vec_env = VecMonitor(vec_env)

hidden_dim = 32
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
)

N = 4000000
model = CrossQ("MlpPolicy", vec_env, buffer_size=N, verbose=1, tensorboard_log="./tensorboard/", gradient_steps=1) #, train_freq=10)
model.learn(total_timesteps=N)

while True:
    obs = vec_env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render(mode="human")
        time.sleep(0.01)
