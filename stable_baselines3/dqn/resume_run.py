#!/usr/bin/env python3
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import ale_py
import gymnasium as gym
import torch

from stable_baselines3.dqn.dqn_cop import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


def setup_file_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"resume_logger_{log_file.parent.name}_{datetime.now().timestamp()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def make_env(env_id: str, rank: int, seed: int = 0, monitor_dir: Optional[str] = None, atari: bool = False):
    def _init():
        env = gym.make(env_id)
        # Recreate the original wrapper order used by the training script
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        env = AtariWrapper(env) if atari else env
        env.reset(seed=seed + rank)
        return env
    return _init


def setup_env(config: Dict[str, Any], seed: int, monitor_dir: Optional[str] = None):
    env_id = config['env']['id']
    n_envs = config['env']['n_envs']
    atari = config['env']['atari']

    env = DummyVecEnv([
        make_env(env_id, i, seed, monitor_dir, atari)
        for i in range(n_envs)
    ])

    stack = config['env']['n_stack']
    if stack != 0:
        env = VecFrameStack(env, n_stack=stack)

    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Resume one SB3 DQN run into a new child resume folder.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the base run folder, e.g. .../dqn_baseline_run_0")
    parser.add_argument("--extra_timesteps", type=int, required=True,
                        help="Number of additional timesteps to train")
    parser.add_argument("--device", type=str, default="auto",
                        help='Device to use, e.g. "auto", "cpu", "cuda", or "cuda:0"')
    parser.add_argument("--no_replay_buffer", action="store_true",
                        help="Do not load the saved replay buffer")
    parser.add_argument("--resume_name", type=str, default=None,
                        help='Optional explicit name for the resume folder. Default: resume_YYYYMMDD_HHMMSS')
    return parser.parse_args()


def resolve_base_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if run_dir.name.startswith("resume_"):
        return run_dir.parent
    return run_dir


def is_resume_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("resume_")


def find_latest_segment(base_run_dir: Path) -> Path:
    resume_dirs = sorted([p for p in base_run_dir.iterdir() if is_resume_dir(p)], key=lambda p: p.name)
    return resume_dirs[-1] if resume_dirs else base_run_dir


def infer_experiment_root(base_run_dir: Path) -> Path:
    experiment_root = base_run_dir.parent
    config_path = experiment_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at expected location: {config_path}")
    return experiment_root


def load_config(experiment_root: Path) -> Dict[str, Any]:
    config_path = experiment_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_config_name_and_run_index(base_run_dir: Path) -> Tuple[str, int]:
    m = re.match(r"(.+)_run_(\d+)$", base_run_dir.name)
    if not m:
        raise ValueError(
            f'Could not infer config name and run index from folder name "{base_run_dir.name}". '
            f'Expected pattern like "dqn_baseline_run_0".'
        )
    return m.group(1), int(m.group(2))


def recover_seed_from_config(config: Dict[str, Any], target_config_name: str, target_run_index: int) -> int:
    all_configs: List[Tuple[str, int]] = []
    for config_name in config['training']['configurations']:
        for seed in config['training']['seeds']:
            all_configs.append((config_name, seed))

    if target_run_index < 0 or target_run_index >= len(all_configs):
        raise IndexError(
            f"Run index {target_run_index} is out of range for the config mapping "
            f"(total mapped runs: {len(all_configs)})."
        )

    mapped_config_name, mapped_seed = all_configs[target_run_index]
    if mapped_config_name != target_config_name:
        raise ValueError(
            f'Run folder says config "{target_config_name}" at index {target_run_index}, '
            f'but config.yaml maps that index to "{mapped_config_name}".'
        )
    return mapped_seed


def choose_resume_name(explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    return f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_callbacks(config: Dict[str, Any], eval_env, resume_dir: Path):
    actual_eval_freq = config['logging']['eval_freq'] // config['env'].get('n_envs', 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(resume_dir),
        log_path=str(resume_dir),
        eval_freq=actual_eval_freq,
        n_eval_episodes=config['logging']['n_eval_episodes'],
        deterministic=True,
        verbose=config['logging'].get('verbose', 0),
    )

    if config['logging']['model_save_freq'] != 0:
        actual_model_save_freq = config['logging']['model_save_freq'] // config['env'].get('n_envs', 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=actual_model_save_freq,
            save_replay_buffer=config['logging']['save_buffer'],
            save_path=str(resume_dir),
            name_prefix="check_dip",
        )
        return CallbackList([checkpoint_callback, eval_callback])

    return eval_callback


def save_yaml(path: Path, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    args = parse_args()
    base_run_dir = resolve_base_run_dir(Path(args.run_dir))
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {base_run_dir}")

    experiment_root = infer_experiment_root(base_run_dir)
    config = load_config(experiment_root)

    config_name, run_index = infer_config_name_and_run_index(base_run_dir)
    seed = recover_seed_from_config(config, config_name, run_index)

    source_segment_dir = find_latest_segment(base_run_dir)
    source_model_path = source_segment_dir / "final_model.zip"
    source_buffer_path = source_segment_dir / "final_model.pkl"

    if not source_model_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_model_path}")
    if (not args.no_replay_buffer) and (not source_buffer_path.exists()):
        raise FileNotFoundError(
            f"Replay buffer requested but not found: {source_buffer_path}\n"
            f"Pass --no_replay_buffer if you really want to resume without it."
        )

    resume_dir = base_run_dir / choose_resume_name(args.resume_name)
    if resume_dir.exists():
        raise FileExistsError(f"Resume directory already exists: {resume_dir}")

    # Create output structure for the resumed segment
    train_monitor_dir = resume_dir / "monitor" / "train"
    eval_monitor_dir = resume_dir / "monitor" / "eval"
    train_monitor_dir.mkdir(parents=True, exist_ok=False)
    eval_monitor_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_file_logger(resume_dir / "train_log.txt")
    logger.info("Starting resume run")
    logger.info(f"Base run dir: {base_run_dir}")
    logger.info(f"Source segment: {source_segment_dir}")
    logger.info(f"Resume output dir: {resume_dir}")
    logger.info(f"Config name: {config_name}")
    logger.info(f"Recovered seed: {seed}")

    env = setup_env(config, seed, str(train_monitor_dir))
    eval_env = setup_env(config, seed + 1000, str(eval_monitor_dir))
    eval_env = VecTransposeImage(eval_env) if config['env']['atari'] else eval_env

    callbacks = build_callbacks(config, eval_env, resume_dir)

    sb3_logger = configure(
        folder=str(resume_dir),
        format_strings=config['logging']['format']
    )

    logger.info(f"Loading model from: {source_model_path}")
    model = DQN.load(
        path=str(source_model_path),
        env=env,
        device=args.device,
    )
    model.set_logger(sb3_logger)

    if not args.no_replay_buffer:
        logger.info(f"Loading replay buffer from: {source_buffer_path}")
        model.load_replay_buffer(str(source_buffer_path))
    else:
        logger.info("Skipping replay buffer load because --no_replay_buffer was set")

    previous_num_timesteps = int(model.num_timesteps)
    logger.info(f"Recovered num_timesteps from loaded model: {previous_num_timesteps}")

    resume_metadata = {
        "base_run_dir": str(base_run_dir),
        "source_segment_dir": str(source_segment_dir),
        "source_model_path": str(source_model_path),
        "source_replay_buffer_path": None if args.no_replay_buffer else str(source_buffer_path),
        "resume_dir": str(resume_dir),
        "resume_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_name": config_name,
        "run_index": run_index,
        "seed": seed,
        "device": args.device,
        "extra_timesteps_requested": int(args.extra_timesteps),
        "previous_num_timesteps": previous_num_timesteps,
        "load_replay_buffer": not args.no_replay_buffer,
        "parent_segment_name": source_segment_dir.name,
        "base_segment_name": base_run_dir.name,
    }
    save_yaml(resume_dir / "resume_metadata.yaml", resume_metadata)

    try:
        model.learn(
            total_timesteps=int(args.extra_timesteps),
            callback=callbacks,
            progress_bar=config['logging'].get('progress_bar', False),
            reset_num_timesteps=False,
        )

        final_model_path = resume_dir / "final_model"
        logger.info(f"Saving resumed final model to: {final_model_path}.zip")
        model.save(str(final_model_path))
        model.save_replay_buffer(str(final_model_path))

        final_metadata = dict(resume_metadata)
        final_metadata["completed"] = True
        final_metadata["final_num_timesteps"] = int(model.num_timesteps)
        final_metadata["timesteps_added"] = int(model.num_timesteps) - previous_num_timesteps
        save_yaml(resume_dir / "resume_metadata.yaml", final_metadata)

        logger.info("Resume training completed successfully")

    except Exception as e:
        logger.exception(f"Resume training failed: {e}")
        fail_metadata = dict(resume_metadata)
        fail_metadata["completed"] = False
        fail_metadata["error"] = str(e)
        fail_metadata["num_timesteps_when_failed"] = int(getattr(model, "num_timesteps", previous_num_timesteps))
        save_yaml(resume_dir / "resume_metadata.yaml", fail_metadata)
        raise

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Resume interrupted by user. Exiting gracefully.")
        sys.exit(0)
