import gymnasium as gym  
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
import numpy as np
import pygame

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "Taxi Env",
    "Algorithm": "DQN"
}
run = wandb.init(
    project="RL-taxi-DQN",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

env= gym.make("Taxi-v3", render_mode="human")

#check_env(env)
state, action =env.reset()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500, callback=WandbCallback(gradient_save_freq=100,
        model_save_path=f"models/{run.id}", verbose=1,))
# model.learn(total_timesteps=500)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")