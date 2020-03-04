from stable_baselines.common.env_checker import check_env

from gym_text2048.envs import Text2048Env, Text2048CappedEnv, Text2048WithHeuristicEnv, Text2048CappedWithHeuristicEnv

if __name__ == "__main__":
    envs = [
        Text2048Env(),
        Text2048CappedEnv(),
        Text2048WithHeuristicEnv(),
        Text2048CappedWithHeuristicEnv(),
    ]

    for env in envs:
        check_env(env)
