from contextlib import closing
from itertools import product, tee
import logging
import sys

import gym
from gym import error, spaces
from gym.utils import colorize, seeding
import numpy as np
from six import StringIO, b

import tensorflow as tf

logger = logging.getLogger(__name__)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

TILE_FORMAT = {
    1: ('white', True),
    2: ('cyan', False),
    3: ('cyan', True),
    4: ('blue', False),
    5: ('blue', True),
    6: ('magenta', False),
    7: ('magenta', True),
    8: ('red', False),
    9: ('red', True),
    10: ('yellow', False),
    11: ('yellow', True),
}


class Text2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=4, one_hot=False, cnn=False):
        self.size = size
        self._one_hot = one_hot
        self._cnn = cnn

        self.action_space = spaces.Discrete(4)
        if one_hot:
            self.observation_space = spaces.MultiBinary(size*size*16)
        elif cnn:
            self.observation_space = spaces.Box(0, 1, [size,size,16], dtype=int)
        else:
            self.observation_space = spaces.Box(0, 16, [size,size], dtype=int)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _add_random_tile(self):
        empty_tiles = [t for t in product(*tee(range(self.size)))
                      if self.board[t] == 0]
        if len(empty_tiles) > 0:
            k = self.np_random.randint(len(empty_tiles))
            value = 1 if self.np_random.random() < 0.9 else 2
            self.board[empty_tiles[k]] = value

    def _compress(self, view):
        changed = False
        for j in range(self.size):
            count = 0
            for i in range(self.size):
                if view[i][j] != 0:
                    view[count][j], view[i][j] = view[i][j], view[count][j]
                    if count != i:
                        changed = True
                    count += 1
        return changed

    def _merge(self, view):
        reward = 0
        for j in range(self.size):
            for i in range(self.size - 1):
                if view[i][j] == view[i + 1][j] and view[i][j] != 0:
                    view[i][j] += 1
                    view[i + 1][j] = 0
                    reward += (2 ** view[i][j])
        return reward

    def _is_done(self):
        def can_merge(i, j):
            return any([self.board[i][j] == self.board[i + di][j + dj]
                        for di, dj in ((-1, 0), (0, -1), (0, 1), (1, 0))
                        if (i + di < self.size and i + di >= 0) and
                           (j + dj < self.size and j + dj >= 0)])

        for i, j in product(*tee(range(self.size))):
            if self.board[i][j] == 0 or can_merge(i, j):
                return False
        return True

    def _get_reward(self):
        return self.last_action_score

    def _get_board(self):
        if self._one_hot:
            return np.ravel(np.eye(16, dtype=int)[self.board])
        if self._cnn:
            return (np.eye(16, dtype=int)[self.board])
            # return self.board.reshape(self.size, self.size, 1)
        return self.board

    def step(self, action):
        assert self.action_space.contains(action)

        view = np.rot90(self.board, k=action)
        changed = self._compress(view)
        action_score = self._merge(view)
        if changed or action_score > 0:
            self._compress(view)
            self._add_random_tile()
        else:
            self._invalid_count += 1
            return self._get_board(), -32, self._invalid_count >= 10, {'score': self.score}

        self.last_action = action
        self.last_action_score = action_score
        self.score += action_score

        return self._get_board(), self._get_reward(), self._is_done(), {'score': self.score}

    def reset(self):
        self.score = 0
        self.last_action = None
        self.last_action_score = 0
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._add_random_tile()
        self._add_random_tile()
        self._invalid_count = 0
        return self._get_board()

    def render(self, mode='human'):
        out = StringIO if mode == 'ansi' else sys.stdout

        if self.last_action is not None:
            out.write("  ({})\n".format(["Up", "Right", "Down", "Left"][self.last_action]))

        def tile_to_symbol(tile):
            if tile == 0:
                return "     "
            elif tile in TILE_FORMAT:
                return colorize("{:>5}".format(2 ** tile), *TILE_FORMAT[tile])
            return colorize("{:>5}".format(2 ** tile), "gray", False)

        hline = "\n|" + "|".join(["-----"] * len(self.board)) + "|\n"
        symbols = [[tile_to_symbol(t) for t in line] for line in self.board]
        out.write(hline +
                  hline.join('|' + '|'.join(line) + '|' for line in symbols) +
                  hline)

        if mode != 'human':
            with closing(out):
                return out.getvalue()
