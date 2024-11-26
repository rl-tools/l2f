from l2f.wrappers.sb3 import L2F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import time
import torch

library = "sbx"
algorithm = "sac"

seed = 1338

if library == "sbx":
    from sbx import PPO, SAC, TD3
elif library == "sb3":
    from stable_baselines3 import PPO, SAC, TD3

if algorithm == "ppo":
    n_envs = 64
    vec_env = L2F(n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)

    hidden_dim = 64
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
    )

    N = 40000000
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=1, n_epochs=1, n_steps=128,batch_size = 4096, policy_kwargs=policy_kwargs, normalize_advantage=True, learning_rate=1e-3, vf_coef=1, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=N)
elif algorithm == "sac":
    n_envs = 16
    vec_env = L2F(n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)

    hidden_dim = 32
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
    )

    N = 4000000
    model = SAC("MlpPolicy", vec_env, seed=seed, buffer_size=N, verbose=1, learning_starts=10000, tensorboard_log="./tensorboard/", gradient_steps=1, train_freq=16) #, train_freq=10)
    model.learn(total_timesteps=N)
elif algorithm == "td3":
    n_envs = 1
    vec_env = L2F(n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)

    hidden_dim = 32
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim], qf=[hidden_dim, hidden_dim]),
    )

    N = 4000000
    model = TD3("MlpPolicy", vec_env, seed=seed, buffer_size=N, verbose=1, learning_starts=10000, tensorboard_log="./tensorboard/", gradient_steps=1, train_freq=16)
    model.learn(total_timesteps=N)
else:
    raise ValueError(f"Unknown algorithm {algorithm}")


obs = vec_env.reset()
episode_return = 0
episode_step = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render(mode="human")
    time.sleep(0.01)
    episode_return += rewards[0]
    episode_step += 1
    if episode_step > 500 or dones[0]:
        print(f"Episode done in {episode_step} steps with return {episode_return}")
        episode_step = 0
        episode_return = 0
        obs = vec_env.reset()
