import Env1
import render as render
import torch
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_checker import check_env

env= Env1.HuntingEnv() # initialize the custom environment class
#check_env(env)
state, action =env.reset()

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
obs, info = env.reset()
while True:
    env.render()
    action, _states = model.predict(obs, deterministic=True)

    #print(type(action))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated == True:
        obs, info = env.reset()

env.close()