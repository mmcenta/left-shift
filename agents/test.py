import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import random

from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import conv, conv_to_fc, linear

def extract_max_tile(obs):
    max_val = 0
    for i in range(4):
        for j in range(4):
            max_val = max(max_val, np.argmax(obs[i,j]))
    return max_val

def print_histogram(hist):
    print('Histogram of maximum tile achieved:')
    for i in range(15):
        if hist[i] > 0:
            print(f'{2**i}: {hist[i]}')


histogram = np.zeros(15, dtype=int)
max_val = 0

def custom_callback(_locals, _globals):
    '''
    Custom callback
    :param _locals: (dict)
    :param _globals: (dict)
    '''
    timestep = _locals['self'].num_timesteps
    global histogram
    global max_val
    if _locals['reset']:
        histogram[max_val] += 1
    max_val = extract_max_tile(_locals['obs'])
    if timestep % 500 == 0 :
        print_histogram(histogram)


def evaluate(model, env_id, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    env = DummyVecEnv([lambda: gym.make_env(env_id, cnn=True)])
    all_episode_rewards = []
    max_achieved = np.zeros(15, dtype=int)
    for _ in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, extras = env.step(action)
            if reward < 0:
                action = random.sample(range(4),1)[0]
                obs, reward, done, extras = env.step(action)
        max_achieved[extract_max_tile(obs)] += 1
        all_episode_rewards.append(extras['score'])

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Reward: ", mean_episode_reward)
    print("Max Achieved:", max_achieved)
    print_histogram(max_achieved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', '-mn', type=str,
                        help='Name of the model to test.')
    parser.add_argument('--env', '-e', type=str, default="gym_text2048:Text2048-v0",
                        help='Id of the environment to test the agent.')
    parser.add_argument('--save-dir', '-sd', type=str, default="models",
                        help='Directory in which the agents are saved.')
    parser.add_argument('--n-episodes', '-n', type=int, default=100,
                        help='Number of episodes for which to test the model.')
    args = parser.parse_args()

    model_path = os.path.join(args.save_dir, "{:}.zip".format(args.model_name))
    if os.path.exists(model_path):
        dqn_model = DQN.load(os.path.join(args.save_dir, args.model_name))

    evaluate(dqn_model, args.env, num_episodes=args.n_episodes)
