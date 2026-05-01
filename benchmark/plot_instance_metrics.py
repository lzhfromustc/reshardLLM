#!/usr/bin/env python3
"""Plot GPU utilization and fragmentation proportion (Figure 12) from instance CSV."""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_instance_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    # Map instance_id (UUID or int) to 0,1,2,... for plotting.
    if "instance_id" in df.columns:
        if df["instance_id"].dtype == object or not np.issubdtype(df["instance_id"].dtype, np.integer):
            uniq = df["instance_id"].unique()
            id_to_idx = {u: i for i, u in enumerate(sorted(uniq, key=str))}
            df = df.copy()
            df["instance_id"] = df["instance_id"].map(id_to_idx)
        df["instance_id"] = df["instance_id"].astype(int)
    return df.sort_values("timestamp")


def plot_gpu_utilization(df: pd.DataFrame, output_path: str, instance_num: int | None = None) -> None:
    if instance_num is None:
        instance_num = int(df["instance_id"].max()) + 1 if len(df) else 0
    time_begin = df["timestamp"].min()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(instance_num):
        sub = df[df["instance_id"] == i]
        if sub.empty:
            continue
        ts = sub["timestamp"].to_numpy() - time_begin
        usage = sub["gpu_cache_usage"].to_numpy() * 100
        ax.plot(ts, usage, label=f"instance_{i}", linewidth=0.8)
    ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU cache usage (%)")
    ax.set_title("GPU utilization rate")
    usage_all = df["gpu_cache_usage"].to_numpy() * 100
    u_max = np.nanmax(usage_all) if len(usage_all) else 0
    ax.set_ylim(0, max(2.0, u_max * 1.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fragmentation_proportion_fig12(df: pd.DataFrame, output_path: str, instance_num: int | None = None) -> None:
    if "num_total_gpu_blocks" not in df.columns:
        u = df["gpu_cache_usage"].replace(1.0, 0.9999)
        df = df.copy()
        df["num_total_gpu_blocks"] = (df["num_available_gpu_blocks"] / (1 - u)).round().astype(int)
    if instance_num is None:
        instance_num = int(df["instance_id"].max()) + 1 if len(df) else 0

    timestamps = sorted(df["timestamp"].unique())
    proportions: list[float] = []
    for t in timestamps:
        rows = df[df["timestamp"] == t]
        total_free = rows["num_available_gpu_blocks"].sum()
        total_blocks = rows["num_total_gpu_blocks"].sum()
        if total_blocks <= 0:
            proportions.append(np.nan)
            continue
        hol_demands = []
        for _, r in rows.iterrows():
            if r.get("num_waiting_requests", 0) > 0 and r.get("num_blocks_first_waiting_request", 0) > 0:
                hol_demands.append(int(r["num_blocks_first_waiting_request"]))
        hol_demands.sort()
        fragmented = 0
        for d in hol_demands:
            if fragmented + d <= total_free:
                fragmented += d
            else:
                break
        proportions.append(fragmented / total_blocks)

    time_begin = timestamps[0]
    ts = np.array(timestamps) - time_begin
    proportions = np.array(proportions)
    valid = ~np.isnan(proportions)
    if not np.any(valid):
        proportions = np.zeros_like(proportions)
    else:
        proportions = np.where(valid, proportions, np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, proportions * 100, color="tab:blue", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fragmentation proportion (%)")
    p_max = np.nanmax(proportions) * 100 if np.any(valid) else 0
    ax.set_title("Figure 12: Fragmented memory proportion of cluster total memory")
    ax.set_ylim(0, max(15.0, p_max * 1.15))
    ax.set_xlim(0, 2600)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--instance-num", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.instance_csv):
        print(f"Error: instance CSV not found: {args.instance_csv}")
        return 1

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.instance_csv))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.instance_csv.replace("_instance.csv", "")))[0]

    df = load_instance_csv(args.instance_csv)
    if df.empty:
        print("Error: instance CSV is empty.")
        return 1

    plot_gpu_utilization(df, os.path.join(output_dir, f"{base}_gpu_utilization.png"), instance_num=args.instance_num)
    plot_fragmentation_proportion_fig12(
        df,
        os.path.join(output_dir, f"{base}_fragmentation_proportion_fig12.png"),
        instance_num=args.instance_num,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
