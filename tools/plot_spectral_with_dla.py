#!/usr/bin/env python3
"""Monitor optimum quality for random-target runs (spectral_canonical_results).

Two metrics vs K, for a fixed --dim:
  1. full-DLA fraction, estimated from rejection counts:
         accepts / (accepts + sum(n_rejected))
     over all rows (timed-out rows contribute their rejects, no accept).
  2. fraction of solved states with overlap >= --threshold (default 0.99),
     over the rows that actually produced an overlap (not timed out).
"""

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
        description="Plot full-DLA fraction and %above-threshold by K")
    p.add_argument("--dim", type=int, help="Dimension to filter")
    p.add_argument("--threshold", type=float, default=0.99,
                   help="Overlap threshold for 'solved' (default 0.99)")
    p.add_argument("--out", type=str, default=None,
                   help="Output file (default: auto if headless)")
    return p.parse_args()


def _is_blank(v):
    return v is None or v == ""


def scan_available(results_dir):
    """Return {dim: {K: n_rows}}."""
    combos = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return combos
    for csv_file in results_path.glob("*.csv"):
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                dim = int(row["dim"]); K = int(row["K"])
                combos.setdefault(dim, {}).setdefault(K, 0)
                combos[dim][K] += 1
    return combos


def show_available(combos):
    if not combos:
        print("No data found.")
        return
    print("Available dims (with K coverage):\n")
    for dim in sorted(combos):
        k_info = ", ".join(f"K={k}:{n}" for k, n in sorted(combos[dim].items()))
        print(f"  --dim {dim}    [{k_info}]")


def load_data(results_dir, dim):
    """Per K: accepts, total_rejected, overlaps (list), timeouts.

    accepts        = rows that found a full-DLA subset (have n_rejected, not timed out)
    total_rejected = sum of n_rejected over ALL rows that have it (incl. timed-out)
    overlaps       = overlap values from rows that produced one
    """
    agg = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return agg
    for csv_file in results_path.glob("*.csv"):
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["dim"]) != dim:
                    continue
                K = int(row["K"])
                d = agg.setdefault(K, {"accepts": 0, "rejected": 0,
                                       "overlaps": [], "timeouts": 0})
                timed_out = (str(row.get("timed_out", "")) == "1")
                nr = row.get("n_rejected", "")
                if not _is_blank(nr):
                    d["rejected"] += int(float(nr))
                if not _is_blank(nr) and not timed_out:
                    d["accepts"] += 1
                if timed_out:
                    d["timeouts"] += 1
                ov = row.get("overlap", "")
                if not _is_blank(ov):
                    d["overlaps"].append(float(ov))
    return agg


def plot_data(agg, dim, threshold):
    if not agg:
        print("No data matching dim.")
        return None
    Ks = sorted(agg.keys())

    dla_frac, dla_tot = [], []
    above, n_ov = [], []
    for K in Ks:
        a = agg[K]["accepts"]
        r = agg[K]["rejected"]
        tot = a + r
        dla_frac.append(a / tot if tot > 0 else np.nan)
        dla_tot.append(tot)
        ov = np.array(agg[K]["overlaps"])
        n_ov.append(len(ov))
        above.append(np.mean(ov >= threshold) if len(ov) else np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(Ks) * 0.8), 7),
                                   sharex=True)

    ax1.plot(Ks, dla_frac, "o-")
    ax1.set_ylabel("full-DLA fraction\n(accepts / total sampled)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(f"DLA sampling rate by K (dim={dim})")
    for x, y, n in zip(Ks, dla_frac, dla_tot):
        if not np.isnan(y):
            ax1.annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)

    ax2.plot(Ks, above, "s-", color="tab:green")
    ax2.set_ylabel(f"frac overlap >= {threshold}")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("K")
    ax2.set_xticks(Ks)
    ax2.set_title(f"Optimizer success rate by K (dim={dim})")
    for x, y, n in zip(Ks, above, n_ov):
        if not np.isnan(y):
            ax2.annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)

    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "spectral_canonical_results"

    if args.dim is None:
        show_available(scan_available(results_dir))
        return

    agg = load_data(results_dir, args.dim)
    fig = plot_data(agg, args.dim, args.threshold)
    if fig is None:
        return

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    elif can_connect_wayland() or os.environ.get('DISPLAY'):
        plt.show()
    else:
        out_path = f"overlap_quality_dim{args.dim}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Headless environment detected. Saved to {out_path}")


if __name__ == "__main__":
    main()
