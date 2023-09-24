from gym.envs.registration import register

from environments import DotsAndBoxesEnv

register(
   id='DotsAndBoxes-v0',
   entry_point='env:DotsAndBoxesEnv',
   max_episode_steps=1000,
)

env = DotsAndBoxesEnv(render_mode='human')
num_episodes = 0
max_episodes = 3
env.reset(seed=42)

for agent in env.agent_iter():

   if num_episodes >= max_episodes:
      break

   observation, reward, termination, truncation, info = env.last()

   if termination or truncation:
      env.reset()
      num_episodes += 1
   else:
      # this is where you would insert your policy
      mask = observation["action_mask"]
      action = env.action_space(agent).sample(mask)
      env.step(action)

env.close()
