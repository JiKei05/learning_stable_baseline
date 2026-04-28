#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import pandas as pd


MA_WINDOW = 10


def read_monitor_rewards(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, skiprows=1)
    if "r" not in df.columns:
        raise ValueError(f"Column 'r' not found in {csv_path}")
    return df["r"].reset_index(drop=True)


def add_moving_average(df: pd.DataFrame, reward_col: str = "avg_reward", window: int = MA_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df[f"{reward_col}_ma{window}"] = df[reward_col].rolling(window=window, min_periods=1).mean()
    return df


def average_csv_group(csv_files: List[Path]) -> pd.DataFrame:
    if not csv_files:
        raise ValueError("No CSV files provided for averaging.")

    reward_series = [read_monitor_rewards(p) for p in sorted(csv_files)]
    min_len = min(len(s) for s in reward_series)
    if min_len == 0:
        raise ValueError("At least one CSV file has zero episode rows.")

    truncated = [s.iloc[:min_len].reset_index(drop=True) for s in reward_series]
    reward_matrix = pd.concat(truncated, axis=1)
    reward_matrix.columns = [f"file_{i}" for i in range(len(truncated))]

    df = pd.DataFrame({
        "avg_reward": reward_matrix.mean(axis=1)
    })
    return add_moving_average(df)


def find_eval_csvs_under(folder: Path) -> List[Path]:
    eval_dir = folder / "monitor" / "eval"
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.rglob("*.csv"))


def find_resume_dirs(base_run_dir: Path) -> List[Path]:
    return sorted(
        [p for p in base_run_dir.iterdir() if p.is_dir() and p.name.startswith("resume_")],
        key=lambda p: p.name,
    )


def build_merged_average(base_run_dir: Path) -> pd.DataFrame:
    segments = [base_run_dir] + find_resume_dirs(base_run_dir)
    segment_frames = []

    for segment in segments:
        csv_files = find_eval_csvs_under(segment)
        if not csv_files:
            print(f"Skipping {segment.name}: no eval CSV files found under monitor/eval")
            continue

        averaged = average_csv_group(csv_files)
        averaged["segment"] = segment.name
        averaged["segment_episode"] = range(1, len(averaged) + 1)
        segment_frames.append(averaged)

        print(f"Processed {segment.name}: {len(csv_files)} csv files, {len(averaged)} averaged episodes")

    if not segment_frames:
        raise FileNotFoundError(
            f"No eval monitor CSV files found in base folder or its resume_* children: {base_run_dir}"
        )

    merged = pd.concat(segment_frames, ignore_index=True)
    merged["episode_num"] = range(1, len(merged) + 1)
    merged = merged[["episode_num", "avg_reward", f"avg_reward_ma{MA_WINDOW}", "segment", "segment_episode"]]
    return merged


def find_run_dirs(algorithm_dir: Path) -> List[Path]:
    run_dirs = []
    for p in algorithm_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("resume_"):
            continue
        # Example: dqn_baseline_run_0, dqn_baseline_run_1, ...
        if "_run_" in p.name:
            run_dirs.append(p)
    return sorted(run_dirs, key=lambda p: p.name)



def process_one_run(run_dir: Path, output_name: str = "averaged_eval_merged.csv") -> Path:
    merged = build_merged_average(run_dir)
    output_csv = run_dir / output_name
    merged.to_csv(output_csv, index=False)
    return output_csv


def build_final_average_from_run_averages(run_average_files: List[Path]) -> pd.DataFrame:
    if not run_average_files:
        raise ValueError("No averaged run files provided.")

    series_list = []
    min_len = None

    for csv_file in sorted(run_average_files):
        df = pd.read_csv(csv_file)
        if "avg_reward" not in df.columns:
            raise ValueError(f"Column 'avg_reward' not found in {csv_file}")

        s = df["avg_reward"].reset_index(drop=True)
        series_list.append(s)
        min_len = len(s) if min_len is None else min(min_len, len(s))

    if min_len is None or min_len == 0:
        raise ValueError("At least one averaged run file has zero rows.")

    truncated = [s.iloc[:min_len].reset_index(drop=True) for s in series_list]
    reward_matrix = pd.concat(truncated, axis=1)
    reward_matrix.columns = [f"run_{i}" for i in range(len(truncated))]

    final_df = pd.DataFrame({
        "episode_num": range(1, min_len + 1),
        "avg_reward": reward_matrix.mean(axis=1),
        "std_reward": reward_matrix.std(axis=1),
        "min_reward": reward_matrix.min(axis=1),
        "max_reward": reward_matrix.max(axis=1),
    })
    final_df = add_moving_average(final_df)
    return final_df



def find_algorithm_dirs(root_dir: Path) -> List[Path]:
    """
    Root structure expected:
        root_dir/
            Break_3steps_run_.../
                dqn_baseline_run_0/
                dqn_baseline_run_1/
            Break_double_run_.../
                dqn_baseline_run_0/
                dqn_baseline_run_1/
    """
    algo_dirs = []
    for p in root_dir.iterdir():
        if not p.is_dir():
            continue
        # An algorithm directory is one that contains run dirs like *_run_0
        if any(child.is_dir() and "_run_" in child.name for child in p.iterdir()):
            algo_dirs.append(p)
    return sorted(algo_dirs, key=lambda p: p.name)



def process_one_algorithm_dir(
    algorithm_dir: Path,
    output_name: str = "averaged_eval_merged.csv",
    final_output_name: str = "final_averaged_eval.csv",
) -> Path:
    run_dirs = find_run_dirs(algorithm_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No base run folders found under algorithm directory: {algorithm_dir}")

    created = []
    skipped = []

    for run_dir in run_dirs:
        try:
            output_csv = process_one_run(run_dir, output_name)
            created.append(output_csv)
            print(f"Saved merged averaged CSV to: {output_csv}")
        except FileNotFoundError as e:
            skipped.append((run_dir, str(e)))
            print(f"Skipping {run_dir.name}: {e}")

    if skipped:
        print(f"Skipped {len(skipped)} run folders in {algorithm_dir.name}")
        for run_dir, reason in skipped:
            print(f"  - {run_dir.name}: {reason}")

    if not created:
        raise FileNotFoundError(
            f"No run-level averaged files were created under algorithm directory: {algorithm_dir}"
        )

    final_df = build_final_average_from_run_averages(created)
    final_output_path = algorithm_dir / final_output_name
    final_df.to_csv(final_output_path, index=False)

    print(f"Saved final averaged CSV to: {final_output_path}")
    print(f"Total final averaged episodes for {algorithm_dir.name}: {len(final_df)}")
    return final_output_path



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Given a root folder containing multiple algorithm folders, process each algorithm folder. "
            "Each algorithm folder should contain run folders like dqn_baseline_run_0, and each run folder "
            "should contain monitor/eval CSVs (optionally also inside resume_* subfolders). "
            "The script creates averaged_eval_merged.csv inside each run folder and final_averaged_eval.csv "
            "inside each algorithm folder. A 5-episode moving average column is also added."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to parent folder containing algorithm folders",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="averaged_eval_merged.csv",
        help="Name of the output CSV to create inside each run folder",
    )
    parser.add_argument(
        "--final_output_name",
        type=str,
        default="final_averaged_eval.csv",
        help="Name of the final averaged CSV to create inside each algorithm folder",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    algorithm_dirs = find_algorithm_dirs(root_dir)
    if not algorithm_dirs:
        raise FileNotFoundError(f"No algorithm folders found under: {root_dir}")

    print(f"Found {len(algorithm_dirs)} algorithm folders under {root_dir}")
    for algorithm_dir in algorithm_dirs:
        print(f"\n=== Processing algorithm folder: {algorithm_dir.name} ===")
        process_one_algorithm_dir(
            algorithm_dir,
            output_name=args.output_name,
            final_output_name=args.final_output_name,
        )


if __name__ == "__main__":
    main()
