import numpy as np
import tensorflow.contrib.layers as tf_layers
import tensorflow as tf
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import conv, conv_to_fc, linear

def cnn_custom(image, **kwargs):
    """
    CNN feature extrator for 2048.
    :param image: (TensorFlow Tensor) Image input placeholder.
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN.
    :return: (TensorFlow Tensor) The CNN output layer.
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=128, filter_size=4, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=128, filter_size=3, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=256, filter_size=2, stride=2, pad='VALID', init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=256, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=512, filter_size=2, stride=1, pad='VALID', init_scale=np.sqrt(2), **kwargs))
    layer_lin = conv_to_fc(layer_5)
    return layer_lin

def cnn_5l4(image, **kwargs):
    """
    :param in: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print("image", image)
    layer_1 = activ(conv(image, 'c1', n_filters=222, filter_size=4, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=222, filter_size=2, stride=1, pad='SAME', init_scale=np.sqrt(2), **kwargs))
    layer_lin = conv_to_fc(layer_5)
    return layer_lin

class CustomPolicy(DQNPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
            extractor='', feature_extraction="cnn",
            obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                            n_batch, dueling=dueling, reuse=reuse,
                                            scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if not extractor:
            cnn_extractor = cnn_custom
        elif extractor == '5l4':
            cnn_extractor = cnn_5l4

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                for layer_size in layers:
                    action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                    if layer_norm:
                        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                    action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
