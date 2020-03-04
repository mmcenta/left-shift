import time
import gym

DELAY = .1

env = gym.make('text2048:Text2048-v0')
observation = env.reset()
env.render()
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  time.sleep(DELAY)
  env.render()
  if done:
    print("Done\n")
    break
env.close()