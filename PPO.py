import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MultiInputPolicy
import Env1
import render as render
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

SEED = 42

# Create a single environment for training with SB3
env= Env1.HuntingEnv()
env = TimeLimit(env, max_episode_steps=5)
obs, info = env.reset()


# Option A: use a helper function to create multiple environments
def _make_env():
    """Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper."""
    _env =Env1.HuntingEnv()
    _env = TimeLimit(_env, max_episode_steps=500)
    _env = RolloutInfoWrapper(_env)
    return _env

expert = PPO(
    policy=MultiInputPolicy,
    env=env,
    seed=0,
    batch_size=50,
    ent_coef=0.0,
    learning_rate=0.005,
    n_epochs=100,
    n_steps=200,
)

reward, _ = evaluate_policy(expert, env, 10)
print(f"Reward before training: {reward}")
while True:
    action, _states = expert.predict(obs, deterministic=False)
    #print(type(action))
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated == True:
        obs, info = env.reset()