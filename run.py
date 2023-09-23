import gymnasium as gym
from gym.envs.registration import register
import time

from env import DotsAndBoxesEnv

register(
   id='DotsAndBoxes-v0',
   entry_point='env:DotsAndBoxesEnv',
   max_episode_steps=1000,
)

env = DotsAndBoxesEnv(render_mode='human')
observation, info = env.reset(seed=42)

for _ in range(500):
   env.render()

   # Add a wait time to slow down the rendering
   time.sleep(1)

   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
