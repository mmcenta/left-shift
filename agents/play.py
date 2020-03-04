import gym

ACTION_LOOKUP = {
    'w': 0,
    'd': 1,
    's': 2,
    'a': 3,
}

env = gym.make('text2048:Text2048-v0')
print("Play with WASD\n")
env.reset()
env.render()
for _ in range(1000):
  button = ""
  while button not in ACTION_LOOKUP:
    button = input().strip().lower()
  _, _, done, _ = env.step(ACTION_LOOKUP[button])
  env.render()
  if done:
    print("Done\n")
    break
env.close()