import Env1
import render as render
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import wandb
from wandb.integration.sb3 import WandbCallback
import imageio
import numpy as np
import pygame

# config = {
#     "policy_type": "MultiInputPolicy",
#     "total_timesteps": 25000,
#     "env_name": "Hunting Env",
#     "Algorithm": "A2C"
# }
# run = wandb.init(
#     project="Advanced-Topics-RL",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=True,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )

env= Env1.HuntingEnv() # initialize the custom environment class
#check_env(env)
state, action =env.reset()
model = DQN("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=500, callback=WandbCallback(gradient_save_freq=100,
#         model_save_path=f"models/{run.id}", verbose=1,))
model.learn(total_timesteps=500)
#run.finish()
obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    #print(type(action))
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated == True:
        obs, info = env.reset()
