import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# print(os.getcwd())

# import csv
# from pathlib import Path
# import ale_py

# import gymnasium as gym
# import torch
# from stable_baselines3.dqn.dqn_cop import DQN
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
# from stable_baselines3.common.atari_wrappers import AtariWrapper
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.logger import configure

# # =========================
# # CONFIG
# # =========================
# MODEL_PATH = "./stable_baselines3/dqn/logs/Lunar_ddqn_ogsb3/dqn_baseline_run_3/check_dip_1180000_steps"   # your chosen checkpoint
# ENV_ID = "LunarLander-v3"                         # change to your env
# N_EPISODES = 10
# CSV_PATH = "./stable_baselines3/dqn/logs/Lunar_ddqn_ogsb3/dqn_baseline_run_3/check_model_csv/1180000_steps.csv"
# DETERMINISTIC = True                            # best for checkpoint comparison
# # =========================

# def get_q_values(model, obs):
#     """
#     Return Q-values as a 1D numpy array for a single observation.
#     """
#     obs_tensor, _ = model.policy.obs_to_tensor(obs)
#     with torch.no_grad():
#         q_values = model.q_net(obs_tensor)
#     return q_values.cpu().numpy()[0]    


# def make_env(seed: int):
#     """
#     Recreate the evaluation environment.
#     If you used wrappers during training, recreate them here too.
#     """
#     env = gym.make(ENV_ID)
#     env.reset(seed=seed)
#     env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)
#     return env


# def main():
#     # Pick device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     env = make_env(15)
    

#     # Load model onto selected device
#     model = DQN.load(MODEL_PATH, env=env, device=device)

#     results = []

#     for episode in range(1, N_EPISODES + 1):
#         obs = env.reset()
#         done = [False]
#         truncated = [{'TimeLimit.truncated': False}]
#         episode_reward = 0.0
#         episode_length = 0
#         step_idx = 0


#         while not (done[0] or truncated[0]['TimeLimit.truncated']):
#             q_values = get_q_values(model, obs)
#             action, _ = model.predict(obs, deterministic=DETERMINISTIC)
#             obs, reward, done, truncated = env.step(action)

#             episode_reward += float(reward[0])
#             step_idx += 1

#             row = {
#                 "episode": episode,
#                 "step": step_idx,
#                 "action": action,
#                 "reward": float(reward),
#                 "done": bool(done),
#                 "truncated": bool(truncated),
#             }


#             # q-value columns
#             for i, q in enumerate(q_values):
#                 row[f"q_{i}"] = float(q)

#             results.append(row)


#         print(
#             f"Episode {episode:02d} | reward = {episode_reward:8.2f} | length = {step_idx}"
#         )

#     env.close()

#     # Build header dynamically from first row
#     Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
#     fieldnames = list(results[0].keys())

#     with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)

#     print(f"Saved rollout log to {CSV_PATH}")


# if __name__ == "__main__":
#     main()

import csv
import argparse
from pathlib import Path
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an SB3 DQN model for N episodes and save raw episode rewards to CSV."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model zip file, e.g. ./best_model.zip",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to save the CSV results, e.g. ./evaluation_results.csv",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="BreakoutNoFrameskip-v4",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--n_stack",
        type=int,
        default=4,
        help="Number of stacked frames",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during evaluation",
    )
    return parser.parse_args()


def make_env(env_id: str):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)

        # Keep Atari preprocessing, but do NOT clip rewards
        env = AtariWrapper(
            env,
            clip_reward=False,
            terminal_on_life_loss=False,
        )
        return env

    return _init


def main():
    args = parse_args()
    print("Starting")

    env = DummyVecEnv([make_env(args.env_id)])
    env = VecFrameStack(env, n_stack=args.n_stack)

    model = DQN.load(args.model_path, env=env)
    print("loaded model")

    results = []

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_idx = 0

    while episode_idx < args.n_episodes:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, infos = env.step(action)

        reward = float(rewards[0])
        done = bool(dones[0])

        episode_reward += reward
        episode_length += 1

        if done:
            results.append(
                {
                    "episode": episode_idx + 1,
                    "reward": episode_reward,
                    "length": episode_length,
                }
            )

            print(
                f"Episode {episode_idx + 1}: "
                f"reward={episode_reward:.2f}, length={episode_length}"
            )

            episode_idx += 1
            episode_reward = 0.0
            episode_length = 0
            obs = env.reset()

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length"])
        writer.writeheader()
        writer.writerows(results)

    all_rewards = [r["reward"] for r in results]
    avg_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    print("\n===== Evaluation Summary =====")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()