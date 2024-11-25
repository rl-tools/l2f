from l2f.wrappers.sb3 import L2F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import time
import torch



n_envs = 64
vec_env = L2F(n_envs)
vec_env = VecMonitor(vec_env)

hidden_dim = 64
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
)

N = 40000000
model = PPO("MlpPolicy", vec_env, verbose=1, n_epochs=1, n_steps=128,batch_size = 4096, policy_kwargs=policy_kwargs, normalize_advantage=True, learning_rate=1e-3, vf_coef=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=N)

while True:
    obs = vec_env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render(mode="human")
        time.sleep(0.01)
