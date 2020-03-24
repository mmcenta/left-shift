import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse

def plot(file_path):
  f = np.load(file_path)
  timesteps = f['timesteps']
  results = f['results']
  ep_lengths = f['ep_lengths']
  mean_reward = np.mean(results, axis=1).flatten()
  mean_length = np.mean(ep_lengths, axis=1).flatten()

  millions = lambda x, pos: '%1.1f M' % (x/1e6)
  formatter = FuncFormatter(millions)

  plt.subplot(2,1,1)
  ax = plt.gca()
  ax.xaxis.set_major_formatter(formatter)

  plt.plot(timesteps, mean_reward)
  plt.xlabel('timesteps')
  plt.ylabel('score')

  plt.subplot(2,1,2)
  ax = plt.gca()
  ax.xaxis.set_major_formatter(formatter)

  plt.plot(timesteps, mean_length)
  plt.xlabel('timesteps')
  plt.ylabel('episode length')

  plt.show()

  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file', help='Path to log file')
  args = parser.parse_args()

  plot(args.file)
