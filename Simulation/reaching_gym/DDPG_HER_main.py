# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:27:23 2021

@author: 14488
"""

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from ur5_symbol_v1 import ur5_symbol_v1
import gym

env = ur5_symbol_v1()
model_class = DDPG  # works also with SAC, DDPG and TD3
N_BITS = 15

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=1,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=0,
)

# Train the model
model.learn(10000)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load('./her_bit_env', env=env)

obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()