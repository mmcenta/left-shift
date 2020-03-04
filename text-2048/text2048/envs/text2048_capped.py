import logging

from gym import spaces
import numpy as np

from text2048.envs import Text2048Env


logger = logging.getLogger(__name__)


class Text2048CappedEnv(Text2048Env):
    def __init__(self, size=4, goal_tile=11):
        super(Text2048CappedEnv, self).__init__(size=size)
        self.goal_tile = goal_tile
        self.observation_space = spaces.MultiDiscrete([goal_tile] * size * size)

    def _is_done(self):
        max_tile = np.max(self.board)
        return (max_tile >= self.goal_tile or
                super(Text2048CappedEnv, self)._is_done())
