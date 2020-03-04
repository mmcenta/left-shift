import logging

import numpy as np

from text2048.envs import Text2048Env


logger = logging.getLogger(__name__)


class Text2048WithHeuristicEnv(Text2048Env):
    def __init__(self, size=4, merge_weight=0., empty_weight=0.,
                 monotonicity_weight=0., monotonicity_exp=1.,
                 sum_weight=0., sum_exp=1.):
        self.merge_weight = merge_weight
        self.empty_weight = empty_weight
        self.monotonicity_weight = monotonicity_weight
        self.monotonicity_exp = monotonicity_exp
        self.sum_weight = sum_weight
        self.sum_exp = sum_exp
        super(Text2048WithHeuristicEnv, self).__init__(size=size)

    def _calculate_state_value(self):
        # Sum of the logs of the tile values
        tile_sum = np.sum(np.power(self.board, self.sum_exp))

        # Count zeros on the board
        empty = self.size * self.size - np.count_nonzero(self.board)

        # Count possible merges on the board
        def count_merges(line):
            count, merge, prev = 0, 0, 0
            for value in line:
                if value != 0:
                    if value == prev:
                        count += 1
                    elif count > 1:
                        merge += 1 + count
                        count = 0
            return merge

        # NOTE: maybe try taking the maximum of the lines and columns instead
        merges = sum([count_merges(self.board[i]) for i in range(self.size)])
        merges += sum([count_merges(self.board[:][j]) for j in range(self.size)])

        # Score the monotonicity of each row and column
        def score_monotonicity(line):
            left, right = 0., 0.
            for i in range(self.size):
                if line[i-1] > line[i]:
                    left += (pow(line[i-1], self.monotonicity_exp) -
                             pow(line[i], self.monotonicity_exp))
                else:
                    right += (pow(line[i], self.monotonicity_exp) -
                                           pow(line[i-1], self.monotonicity_exp))
            # NOTE: original code from github.com/nneonneo/2048-ai/ uses min
            # instead of max. This doesn't seem to reward the correct behaviour
            return max(left, right)

        monotonicity = sum([score_monotonicity(self.board[i]) for i in range(self.size)])
        monotonicity += sum([score_monotonicity(self.board[:][j]) for j in range(self.size)])

        # Return weighted sum of heuristic scores
        return (self.empty_weight * empty +
                self.merge_weight * merges -
                self.monotonicity_weight * monotonicity -
                self.sum_weight * tile_sum)

    def _get_reward(self):
        curr_value = self._calculate_state_value()
        prev_value = self.last_state_value
        self.last_state_value = curr_value
        return curr_value - prev_value

    def reset(self):
        obs = super(Text2048WithHeuristicEnv, self).reset()
        self.last_state_value = self._calculate_state_value()
        return obs
