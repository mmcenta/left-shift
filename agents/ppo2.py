import argparse
import copy
import os
import random
import time

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common import make_vec_env
import tensorflow as tf
import yaml

from callback import CustomCallbackPPO2
from custom_policy import cnn_5l4, cnn_custom

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    print('Evaluating model.')
    env = model.get_env()
    all_episode_rewards = []
    hist = np.zeros(15, dtype=int)
    for _ in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, extras = env.step(action)
            if reward < 0:
                action = random.sample(range(4),1)[0]
                obs, reward, done, extras = env.step(action)
        hist[env.maximum_tile()] += 1
        all_episode_rewards.append(extras['score'])

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Reward: ", mean_episode_reward)
    print('Histogram of maximum tile achieved:')
    for i in range(1,15):
        if hist[i] > 0:
            print(f'{2**i}: {hist[i]}')


def create_model(hyperparams, env="gym_text2048:Text2048-v0", tensorboard_log='', verbose=-1, seed=0, env_kwargs={}, extractor=''):
    """
    Create a DQN model from parameters. The model always uses the Prioritized
    Replay and Double-Q Learning extensions.
    :param hyperparams: (dict) A dict containing the model hyperparameters.
    :param env: (str) Environment id.
    :param tensorboard_log: (str) The Tensorboard log directory.
    :param verbose: (int) Verbose mode (0: no output, 1: INFO).
    :param seed: (int) Random generator seed.
    :return: (BaseRLModel object) The corresponding model.
    """
    feature_extraction = "cnn" if hyperparams['cnn'] else "mlp"

    if not extractor:
        cnn_extractor = cnn_custom
    elif extractor == '5l4':
        cnn_extractor = cnn_5l4
    

    # Prepare kwargs for the constructor
    policy_kwargs = dict(layers=hyperparams['layers'], cnn_extractor=cnn_extractor)
    model_kwargs = copy.deepcopy(hyperparams)
    del model_kwargs['layers']
    del model_kwargs['cnn']
    del model_kwargs['ln']

    model = PPO2(CnnPolicy,
                # Monitor(gym.make(env, **env_kwargs), None),
                make_vec_env(env, 16, env_kwargs=env_kwargs),
                policy_kwargs=policy_kwargs,
                seed=seed,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                **model_kwargs)

    return model

def get_model(model_name, hyperparams, env, verbose=1, seed=0, env_kwargs={}, tensorboard_log='', extractor=''):
    
    if verbose > 0:
        print("Creating model.")
    save_dir = 'models'
    model_path = os.path.join(save_dir, f'{model_name}.zip')
    if model_name and os.path.exists(model_path):
        model = PPO2.load(model_path)
        env = gym.make(env, **env_kwargs)
        model.set_env(env)
        return model
    else:
        return create_model(hyperparams,
                             tensorboard_log=tensorboard_log,
                             verbose=verbose,
                             seed=seed,
                             env_kwargs=env_kwargs,
                             extractor=extractor)


def train(model, model_name, hyperparams,
          env="gym_text2048:Text2048-v0",
          verbose=-1,
          seed=0,
          n_timesteps=1e7,
          log_interval=1e3,
          log_dir='logs',
          save_freq=1e4,
          save_dir='models',
          hist_freq=100,
          eval_episodes=100,
          env_kwargs={}):
    """
    Create (or load) and train a DQN model. The model always uses the Prioritized
    Replay and Double-Q Learning extensions.
    :param hyperparams: (dict) A dict containing the model hyperparameters.
    :param env: (str) Environment id.
    :param tensorboard_log: (str) The Tensorboard log directory.
    :param verbose: (int) Verbose mode (0: no output, 1: INFO).
    :param seed: (int) Random generator seed.
    :param n_timesteps: (int) Number of timesteps.
    :param log_interval: (int) Log interval.
    :param log_dir: (str) Log directory.
    :param save_freq: (int) Save the model every save_freq steps (if negative, no checkpoint).
    :param save_dir: (str) Save directory.
    :param eval_freq: (int) Evaluate the agent every n steps (if negative, no evaluation).
    :param eval_episodes: (int) Number of episodes to use for evaluation.
    """

    callbacks = []
    if save_freq > 0:
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir,
                                            name_prefix=model_name, verbose=1))

    # if eval_freq > 0:
    #     if verbose > 0:
    #         print("Creating evaluation environment")

    #     env = gym.make(env, **env_kwargs)
    #     env.seed(seed)
    #     eval_callback = EvalCallback(env, best_model_save_path=save_dir,
    #                                 n_eval_episodes=eval_episodes, eval_freq=eval_freq,
    #                                 log_path=log_dir)
    #     callbacks.append(eval_callback)

    if hist_freq > 0:
        custom_callback = CustomCallbackPPO2(log_dir=log_dir, hist_freq=hist_freq, verbose=verbose, log_file=model_name, eval_episodes=eval_episodes)
        callbacks.append(custom_callback)

    if verbose > 0:
        print("Beginning training.")
    try:
        model.learn(total_timesteps=n_timesteps,
                    log_interval=log_interval,
                    callback=callbacks)
    except KeyboardInterrupt:
        pass

    if hist_freq > 0:
        custom_callback._dump_values()

    if model_name:
        if verbose > 0:
            print('Saving final model.')
        model.save(os.path.join(save_dir, model_name))
        if verbose > 0:
            print('Final model saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="gym_text2048:Text2048-v0",
                        help='Environment id.')
    parser.add_argument('--tensorboard-log', '-tb', type=str, default='',
                        help='Tensorboard log directory.')
    parser.add_argument('--hyperparams-file', '-hf', type=str, default='hyperparams/ppo2.yaml',
                        help='Hyperparameter YAML file location.')
    parser.add_argument('--model-name', '-mn', type=str, default='',
                        help='Model name (if it already exists, training will be resumed).')
    parser.add_argument('--n-timesteps', '-n', type=int, default=int(1e7),
                        help='Number of timesteps.')
    parser.add_argument('--log-interval', type=int, default=int(1e4),
                        help='Log interval.')
    parser.add_argument('--hist-freq', type=int, default=100,
                        help='Dumps histogram each n steps.')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of episodes to use for evaluation.')
    parser.add_argument('--save-freq', type=int, default=int(1e5),
                        help='Save the model every n steps (if negative, no checkpoint).')
    parser.add_argument('--save-directory', '-sd', type=str, default='models',
                        help='Save directory.')
    parser.add_argument('--log-directory', '-ld', type=str, default='logs',
                        help='Log directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator seed.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbose mode (0: no output, 1: INFO).')
    parser.add_argument('--no-one-hot', dest='one_hot', action='store_false',
                        help='Disable one-hot encoding')
    parser.add_argument('--no-train', dest='train', action='store_false',
                        help='Disable training')
    parser.add_argument('--eval', dest='eval', action='store_true',
                        help='Enable evaluation')
    parser.add_argument('--extractor', type=str, default='',
                        help='Change extractor')
    args = parser.parse_args()

    # Load hyperparameters
    if args.verbose > 0:
        print("Loading hyperparameters from {:}.".format(args.hyperparams_file))
    with open(args.hyperparams_file, 'r') as f:
        hyperparams = yaml.safe_load(f)

    # Gather train kwargs
    train_kwargs = {
        'env': args.env,
        'verbose': args.verbose,
        'seed': args.seed,
        'n_timesteps': args.n_timesteps,
        'log_interval': args.log_interval,
        'log_dir': args.log_directory,
        'save_freq': args.save_freq,
        'save_dir': args.save_directory,
        'hist_freq': args.hist_freq,
        'eval_episodes': args.eval_episodes,
    }
    env_kwargs = {'one_hot': args.one_hot}
    model = get_model(args.model_name, hyperparams, args.env, verbose=args.verbose, seed=args.seed, env_kwargs=env_kwargs, tensorboard_log = args.tensorboard_log, extractor=args.extractor)

    if args.train:
        train(model, args.model_name, hyperparams, **train_kwargs)
    if args.eval:
        evaluate(model, args.eval_episodes)
