import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Text2048-v0',
    entry_point='text2048.envs:Text2048Env',
    nondeterministic=True,
)
register(
    id='Text2048Capped-v0',
    entry_point='text2048.envs:Text2048CappedEnv',
    nondeterministic=True,
)
register(
    id='Text2048WithHeuristic-v0',
    entry_point='text2048.envs:Text2048WithHeuristicEnv',
    nondeterministic=True,
)
register(
    id='Text2048CappedWithHeuristic-v0',
    entry_point='text2048.envs:Text2048CappedWithHeuristicEnv',
    nondeterministic=True,
)
