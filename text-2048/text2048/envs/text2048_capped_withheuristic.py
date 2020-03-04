import logging

from gym import spaces
import numpy as np

from text2048.envs import Text2048WithHeuristicEnv


logger = logging.getLogger(__name__)


class Text2048CappedWithHeuristicEnv(Text2048WithHeuristicEnv):
    def __init__(self, size=4, goal_tile=11,
                 merge_weight=0., empty_weight=0.,
                 monotonicity_weight=0., monotonicity_exp=1.,
                 sum_weight=0., sum_exp=1.):
        super(Text2048CappedWithHeuristicEnv, self).__init__(size=size,
            merge_weight=merge_weight, empty_weight=empty_weight,
            monotonicity_weight=monotonicity_weight, monotonicity_exp=monotonicity_exp,
            sum_weight=sum_weight, sum_exp=sum_exp)
        self.goal_tile = goal_tile
        self.observation_space = spaces.MultiDiscrete([goal_tile] * size * size)

    def _is_done(self):
        max_tile = np.max(self.board)
        return (max_tile >= self.goal_tile or
                super(Text2048CappedWithHeuristicEnv, self)._is_done())
