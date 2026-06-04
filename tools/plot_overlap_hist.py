#!/usr/bin/env python3
"""Plot found_overlap distributions by K for given dim and n_starts."""

import argparse
import csv
import os
import socket
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def can_connect_wayland():
    runtime = os.environ.get('XDG_RUNTIME_DIR')
    display = os.environ.get('WAYLAND_DISPLAY', 'wayland-0')
    if not runtime:
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(os.path.join(runtime, display))
        s.close()
        return True
    except OSError:
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot found_overlap histograms/violins by K")
    p.add_argument("--dim", type=int, help="Dimension to filter")
    p.add_argument("--n_starts", type=int, help="Number of restarts to filter")
    p.add_argument("--style", choices=["violin", "box", "hist"], default="violin",
                   help="Plot style (default: violin)")
    p.add_argument("--out", type=str, default=None,
                   help="Output file (default: auto-generated if headless)")
    return p.parse_args()


def scan_available(results_dir):
    """Scan all CSVs and return available (dim, n_starts) combinations with counts."""
    combos = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return combos

    for csv_file in results_path.glob("*.csv"):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row["dim"]), int(row["n_starts"]))
                K = int(row["K"])
                combos.setdefault(key, {}).setdefault(K, 0)
                combos[key][K] += 1
    return combos


def show_available(combos):
    """Print available combinations."""
    if not combos:
        print("No data found in explicit_reachability_results/")
        return
    print("Available (dim, n_starts) combinations:\n")
    for (dim, n_starts) in sorted(combos.keys()):
        k_info = ", ".join(f"K={k}:{n}" for k, n in sorted(combos[(dim, n_starts)].items()))
        print(f"  --dim {dim} --n_starts {n_starts}    [{k_info}]")


def load_data(results_dir, dim, n_starts):
    """Load all CSVs matching dim and n_starts, return dict K -> list of found_overlap."""
    data = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return data

    for csv_file in results_path.glob("*.csv"):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["dim"]) != dim:
                    continue
                if int(row["n_starts"]) != n_starts:
                    continue
                K = int(row["K"])
                ov = float(row["found_overlap"])
                data.setdefault(K, []).append(ov)
    return data


def plot_data(data, dim, n_starts, style):
    """Create side-by-side vertical plots for each K."""
    if not data:
        print("No data found matching criteria.")
        return None

    Ks = sorted(data.keys())
    values = [data[K] for K in Ks]

    fig, ax = plt.subplots(figsize=(max(6, len(Ks) * 1.2), 6))

    if style == "violin":
        parts = ax.violinplot(values, positions=range(len(Ks)), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    elif style == "box":
        ax.boxplot(values, positions=range(len(Ks)))
    else:  # hist - use horizontal histograms stacked
        for i, (K, vals) in enumerate(zip(Ks, values)):
            hist, bins = np.histogram(vals, bins=20, range=(0, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            width = 0.8 / max(hist) if max(hist) > 0 else 0.8
            ax.barh(bin_centers, hist * width, left=i - 0.4, height=0.04, alpha=0.7, label=f"K={K}")

    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K={K}\n(n={len(data[K])})" for K in Ks])
    ax.set_ylabel("found_overlap")
    ax.set_title(f"Overlap distribution by K (dim={dim}, n_starts={n_starts})")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "explicit_reachability_results"

    if args.dim is None or args.n_starts is None:
        combos = scan_available(results_dir)
        show_available(combos)
        return

    data = load_data(results_dir, args.dim, args.n_starts)
    fig = plot_data(data, args.dim, args.n_starts, args.style)

    if fig is None:
        return

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    elif can_connect_wayland() or os.environ.get('DISPLAY'):
        plt.show()
    else:
        out_path = f"overlap_dim{args.dim}_nstarts{args.n_starts}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Headless environment detected. Saved to {out_path}")


if __name__ == "__main__":
    main()
