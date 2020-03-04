import gym
import numpy as np
import os
import tensorflow as tf

from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c.utils import conv, conv_to_fc, linear

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for _ in range(num_episodes):
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, extras = env.step(action)

        #env.render()
        all_episode_rewards.append(extras['score'])

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward

def my_cnn(image, **kwargs):
    """
    CNN from Nature paper.
    :param in: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print("image", image)
    layer_1 = activ(conv(image, 'c1', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_lin = conv_to_fc(layer_3)
    # return activ(linear(layer_lin, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    return layer_lin

if __name__ == '__main__':
    tf.enable_eager_execution()

    env_id = "gym_text2048:Text2048-v0"
    env = gym.make(env_id, cnn=True)
    #env = DummyVecEnv([lambda: env, lambda: env, lambda: env])
    #model_name = 'cnn_3l_nothing'
    #model_name = 'cnn_3l_noduel'
    #model_name = 'cnn_3l_nodouble'
    #model_name = 'cnn_3l_prioritized'
    #model_name = 'cnn_3l_default'
    #model_name = 'cnn_3l_noduel_lr'
    #model_name = 'cnn_3l_nodouble_lr'
    #model_name = 'cnn_3l_all'
    #model_name = 'cnn_3l_all_buffersize'
    model_name = 'cnn_5l_double_prioritized_lr'


    if True and os.path.exists(f'{model_name}.zip'):
        dqn_model = DQN.load(model_name)
        dqn_model.set_env(env)
    else:
      pass
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, double_q=False, policy_kwargs={'cnn_extractor': my_cnn, 'dueling': False})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, policy_kwargs={'cnn_extractor': my_cnn, 'dueling': False})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, double_q=False, policy_kwargs={'cnn_extractor': my_cnn})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, double_q=False, prioritized_replay=True, policy_kwargs={'cnn_extractor': my_cnn, 'dueling': False})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, policy_kwargs={'cnn_extractor': my_cnn})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, learning_rate=5e-5, policy_kwargs={'cnn_extractor': my_cnn, 'dueling': False})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, learning_rate=5e-5, double_q=False, policy_kwargs={'cnn_extractor': my_cnn})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, prioritized_replay=True, policy_kwargs={'cnn_extractor': my_cnn})
      #dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, prioritized_replay=True, buffer_size=100000, policy_kwargs={'cnn_extractor': my_cnn})
      dqn_model = DQN('CnnPolicy', env, verbose=1, exploration_final_eps=.1, prioritized_replay=True, policy_kwargs={'cnn_extractor': my_cnn, 'dueling': False}, learning_rate=5e-5)

    dqn_model.learn(total_timesteps=100000, log_interval=10)
    dqn_model.save(model_name)
    mean_reward = evaluate(dqn_model, num_episodes=1000)
