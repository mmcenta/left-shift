import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import argparse


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

def get_data(file_path, num_avg=200):
    f = np.load(file_path)
    results = f['rewards'][:-1]
    ep_lengths = f['lengths']
    max_tiles = f['max_tiles'].astype(int)
    max_tiles = 2**max_tiles
    num_episodes = len(results)
    episodes = np.arange(1, num_episodes+1)

    mean_reward = get_mean(results, num_episodes, num_avg)
    mean_length = get_mean(ep_lengths, num_episodes, num_avg)
    return mean_reward, mean_length
    # millions = lambda x, pos: '%1.1f M' % (x/1e6)
    # formatter = FuncFormatter(millions)

def plot(files, num_avg=200):

    rewards = []
    lengths = []
    episodes = []

    for file in files:
        a, b = get_data(file, num_avg)
        rewards.append(a)
        lengths.append(b)
        num_episodes = len(a)
        episodes.append(np.arange(1, num_episodes+1))

    fid = np.random.randint(1000)
    # plt.subplot(3,1,1)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)
    for episode,reward in zip(episodes,rewards):
        plt.plot(episode, reward)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.savefig(f'fig/score_{fid}.png')
    plt.clf()  

    # plt.subplot(3,1,2)
    # ax = plt.gca()
    # # ax.xaxis.set_major_formatter(formatter)
    # ax.set_yscale('log')
    # ax.set_ylim([1,max(max_tiles)*2])
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    # plt.scatter(episodes, max_tiles, s=5)
    # plt.xlabel('episodes')
    # plt.ylabel('max tile')
    # plt.yticks(2**np.arange(1,12))
    # plt.minorticks_off()
    # plt.savefig(f'fig/maxtile_{fid}.png')    
    # plt.clf()    

    # plt.subplot(3,1,3)
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(formatter)
    for episode,length in zip(episodes,lengths):
        plt.plot(episode, length)
    plt.xlabel('episodes')
    plt.ylabel('episode length')
    plt.savefig(f'fig/eplen_{fid}.png')
    plt.clf()  
    
    # plt.show()
  
if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('file', help='Path to log file')
#   args = parser.parse_args()
    files = ["logs/logs_cnn_5l4.npz", "logs/logs_cnn_5l4_nofc.npz", "logs/logs_cnn_custom.npz", "logs/logs_cnn_custom_nofc.npz"]
    plot(files)
