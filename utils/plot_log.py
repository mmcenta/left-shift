import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse

# def plot(file_path):
#   f = np.load(file_path)
#   timesteps = f['timesteps']
#   results = f['results']
#   ep_lengths = f['ep_lengths']
#   mean_reward = np.mean(results, axis=1).flatten()
#   mean_length = np.mean(ep_lengths, axis=1).flatten()

#   millions = lambda x, pos: '%1.1f M' % (x/1e6)
#   formatter = FuncFormatter(millions)

#   plt.subplot(2,1,1)
#   ax = plt.gca()
#   ax.xaxis.set_major_formatter(formatter)

#   plt.plot(timesteps, mean_reward)
#   plt.xlabel('timesteps')
#   plt.ylabel('score')

#   plt.subplot(2,1,2)
#   ax = plt.gca()
#   ax.xaxis.set_major_formatter(formatter)

#   plt.plot(timesteps, mean_length)
#   plt.xlabel('timesteps')
#   plt.ylabel('episode length')

#   plt.show()

def get_mean(arr, num_episodes, num_avg):
    mean_arr = np.zeros(num_episodes)
    st, cur_sum = 0, 0
    for i in range(num_episodes):
        if i - st == num_avg:
            cur_sum -= arr[st]
            st += 1
        cur_sum += arr[i]
        mean_arr[i] = cur_sum/(i - st  +1)
    return mean_arr

def plot(file_path, num_avg=100):
    f = np.load(file_path)
    results = f['rewards'][:-1]
    ep_lengths = f['lengths']
    max_tiles = f['max_tiles'].astype(int)
    max_tiles = 2**max_tiles
    num_episodes = len(results)
    episodes = np.arange(1, num_episodes+1)

    mean_reward = get_mean(results, num_episodes, num_avg)
    mean_length = get_mean(ep_lengths, num_episodes, num_avg)

    # millions = lambda x, pos: '%1.1f M' % (x/1e6)
    # formatter = FuncFormatter(millions)

    plt.subplot(3,1,1)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)

    plt.plot(episodes, mean_reward)
    plt.xlabel('episodes')
    plt.ylabel('score')

    plt.subplot(3,1,2)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)
    ax.set_yscale('log')
    ax.set_ylim([min(max_tiles)/2,max(max_tiles)*2])

    plt.scatter(episodes, max_tiles)
    plt.xlabel('episodes')
    plt.ylabel('max tile')
    plt.yticks(2**np.arange(1,12))

    plt.subplot(3,1,3)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)

    plt.plot(episodes, mean_length)
    plt.xlabel('episodes')
    plt.ylabel('episode length')
    
    plt.show()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('file', help='Path to log file')
  args = parser.parse_args()

  plot(args.file)
