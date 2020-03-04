# gym-text2048

2048 as an OpenAI Gym environment with a simple text display.

## Environments

This package implements several tasks based on the game 2048. They share the following similarities:

* The **action space** is an instance of `Discrete` with 4 elements, with one for each direction (0 for up, 1 for right, 2 for down and 3 for left);
* The **observation space** is an instance of `MultiDiscrete`, in which each tile of the board is discrete space (the number of elements varies with the board size and whether the task is capped or not). The observations are the logarithmic of base two of the tile values - for example, a tile with value 2048 is represented by an 11 in a observation.

The reward and the end condition depend on the task. Currently, the implemented tasks are:

### Text2048

The Text2048 task models the standard game. The rewards are the sum of the merged tiles in the step (which is the base for the score in the original game). An episode ends only when no more moves are possible. As a result, it is possible to achieve tiles bigger than 2048 - we limit the state space by noting that 2<sup>n<sup>2</sup> + 1</sup> is an upper limit for the value of tiles in a n by n board.

### Text2048WithHeuristic

Similar to Text2048, but instead of the original rewards, it instead uses a heuristic reward. See the Heuristics subsection for more information.

### Text2048Capped

Similar to Text2048, but it stops the game once the maximum tile of the board has reached a value (2048 by default). As a result, the observation space for each tile is limited by the value of the maximum tile.

### Text2048CappedWithHeuristic

This enviroment implements both the custom reward of Text2048WithHeuristic and the end condition of Text2048Capped in the same task.

## Heuristics

A custom reward function is provided on some environments. It is based on the state value heuristics used in [the best 2048 A.I. I could find](https://github.com/nneonneo/2048-ai). The value assigned to a state is a weighted sum of four terms:

* **Empty**: this term is equal to the number of empty of tiles on the board. The only parameter of this term is the weight;
* **Merges**: this term is equal to the number of possible merges on the board. The only parameter of this term is the weight;
* **Monotonicty**: this term rewards rows or columns for having a (eventually partial) monotonic order. Each time it finds a correctly ordered neighbouring tiles, a reward of B<sup>x</sup> - S<sup>y</sup> is given, with B, S and x being the bigger logarithm, the smaller logarithm and the monotonicity exponent parameter, respectively. The maximum of both directions for each line is its value and the board's values is the sum over all rows and columns. It has two parameters: the weight and the exponent.
* **Sum**: this term is the sum of the logarithm of all tiles elevated to a constant power, which is a parameter. It has two parameters: the weight and the exponent.

The reward at a given time step is the difference between the new state value and the previous state value.

## Examples

This repository includes the following examples under the directory `/examples`:

* `play.py` which allows the user to play the game using the WASD keys;
* `random_agent.py` which implements a random agent.

## Installation

Installation can be done locally using `pip`, just download the repository's files and execute the following commands:

```bash
cd gym-text2048
pip install -e .
```
