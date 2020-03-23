import argparse
import os
import random
import time
import yaml

import gym
import numpy as np
from stable_baselines import DQN
from stable_baselines.a2c.utils import conv, conv_to_fc, linear
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy
import tensorflow as tf


def cnn_extractor(scaled_images, **kwargs):
    """
    CNN feature extrator for 2048.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder.
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN.
    :return: (TensorFlow Tensor) The CNN output layer.
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=64, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=128, filter_size=2, stride=2, pad='VALID', init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=128, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))


def create_model(hyperparams, tensorboard_log='', verbose=-1, seed=0, env="gym_text2048:Text2048-v0"):
    """
    Create a DQN model from parameters. The models always uses the Prioritized
    Replay and Double-Q Learning extensions.
    :param params: (dict) A dict containing the model hyperparameters.
    :return: (BaseRLModel object) The corresponding model.
    """
    feature_extraction = "cnn" if hyperparams['cnn'] else "mlp"
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                     reuse=False, obs_phs=None, dueling=True, **_kwargs):
            super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                               n_steps, n_batch, reuse,
                                               feature_extraction=feature_extraction,
                                               obs_phs=obs_phs, dueling=dueling,
                                               layer_norm=hyperparams['ln'], **_kwargs)

    # Delete hyperparameters not supported by the DQN constructor
    del hyperparams['cnn']
    del hyperparams['ln']

    model = DQN(CustomPolicy, env,
                policy_kwargs=hyperparams['layers'],
                prioritized_replay=True,
                dueling=True,
                seed=seed,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                **hyperparams)

    return model


def train(model_name, hyperparams,
          env="gym_text2048:Text2048-v0",
          tensorboard_log='',
          verbose=-1,
          seed=0,
          n_timesteps=1e7,
          log_interval=-1,
          log_dir='logs',
          save_freq=1e4,
          save_dir='models',
          eval_freq=1e4,
          eval_episodes=5):
    if verbose > 0:
        print("Creating model.")
    if os.path.exists(os.path.join(save_dir, f'{model_name}.zip')):
        model = DQN.load(model_name)
    else:
        model = create_model(hyperparams,
                             tensorboard_log=tensorboard_log,
                             verbose=verbose,
                             seed=seed)

    callbacks = []
    if save_freq > 0:
        callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_dir,
                                            name_prefix=model_name, verbose=1))

    if eval_freq > 0:
        if verbose > 0:
            print("Creating evaluation environment")

        env = gym.make(env)
        env.seed(seed)
        eval_callback = EvalCallback(env, best_model_save_path=save_dir,
                                    n_eval_episodes=eval_episodes, eval_freq=eval_freq,
                                    log_path=log_dir)
        callbacks.append(eval_callback)

    if verbose > 0:
        print("Beginning training.")
    try:
        model.learn(total_timesteps=n_timesteps,
                    log_interval=log_interval,
                    callback=callbacks)
    except KeyboardInterrupt:
        pass

    if verbose > 0:
        print('Saving final model.')
    model.save(os.path.join(save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="gym_text2048:Text2048-v0",
                        help='Train environment id.')
    parser.add_argument('-tb', '--tensorboard-log', type=str, default='',
                        help='Tensorboard log directory.')
    parser.add_argument('--hf', '--hyperparams-file', type=str, default='hyperparams/default.yaml',
                        help='Hyperparameter YAML file location.')
    parser.add_argument('-mn', '--model-name', type=str, default='',
                        help='Model name (if it already exists, training will be resumed).')
    parser.add_argument('-n', '--n-timesteps', type=int, default=1e7,
                        help='Number of timesteps')
    parser.add_argument('--log-interval', type=int, default=1e3,
                        help='Log interval.')
    parser.add_argument('--eval-freq', type=int, default=1e4,
                        help='Evaluate the agent every n steps (if negative, no evaluation)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes to use for evaluation')
    parser.add_argument('--save-freq', type=int, default=-1,
                        help='Save the model every n steps (if negative, no checkpoint)')
    parser.add_argument('-sd', '--save-directory', type=str, default='models',
                        help='Save directory.')
    parser.add_argument('-ld', '--log-directory', type=str, default='logs',
                        help='Log directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random generator seed')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbose mode (0: no output, 1: INFO)')
    args = parser.parse_args()

    # Load hyperparameters
    with open(args.hyperparams, 'r') as f:
        hyperparams = yaml.safe_load(f)

    # Gather train kwargs
    train_kwargs = {
        'env': args.env,
        'tensorboard_log': args.tensorboard_log,
        'verbose': args.verbose,
        'seed': args.seed,
        'n_timesteps': args.n_timesteps,
        'log_interval': args.log_interval,
        'log_dir': args.log_directory,
        'save_freq': args.save_freq,
        'save_dir': args.save_directory,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes
    }

    train(args.model_name, hyperparams, **train_kwargs)
