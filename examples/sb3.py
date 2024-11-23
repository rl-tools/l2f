from l2f.wrappers.sb3 import L2F
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import VecMonitor
import time
import torch



vec_env = L2F()
vec_env = VecMonitor(vec_env)

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[32, 32], vf=[32, 32], qf=[32, 32]),
)

# N = 4000000
# model = PPO("MlpPolicy", vec_env, verbose=1, n_epochs=1, n_steps=128, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/")
N = 400000
model = SAC("MlpPolicy", vec_env, buffer_size=N, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=N)

while True:
    obs = vec_env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render(mode="human")
        time.sleep(0.01)
