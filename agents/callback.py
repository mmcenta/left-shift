# unusable until we find how to access states within _on_step()
from stable_baselines.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, extract_tile_fn, print_histogram_fn, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.extract_max_tile = extract_tile_fn
        self.print_histogram = print_histogram_fn
        self.num_episodes = 1
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.histogram = np.zeros(15)
        self.max_val = 0

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        timestep = self.locals['self'].num_timesteps
        num_episodes = len(self.locals['episode_rewards'])
        if num_episodes > self.num_episodes:
            self.num_episodes = num_episodes
            self.histogram[self.max_val] += 1
        print('hey')
        print(self.locals['self'])
        self.max_val = self.extract_max_tile(self.locals['obs'])
        if timestep % 500 == 0 :
            self.print_histogram(self.histogram)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass