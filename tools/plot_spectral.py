#!/usr/bin/env python3
"""Plot DLA-completeness and optimizer unreachability vs K for a fixed --dim.

Three series on one frame:
  - DLA complete:  accepts / (accepts + sum n_rejected)   [check_dla rows]
  - unreachability probability (full DLA): 1 - frac(overlap >= threshold)  [check_dla rows]
  - unreachability probability:            1 - frac(overlap >= threshold)  [no-DLA rows]
Error bars: 95% Wilson score interval (asymmetric).
"""

import argparse
import csv
import os
import socket
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---- font bundle: cmr10 + cm mathtext, silent fallback ----
def setup_font():
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        if "cmr10" in names:
            matplotlib.rcParams["font.family"] = "cmr10"
            matplotlib.rcParams["mathtext.fontset"] = "cm"
            matplotlib.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["axes.formatter.use_mathtext"] = True
            matplotlib.rcParams["font.size"] = 14
    except Exception:
        pass  # silent fallback to defaults


def can_connect_wayland():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    display = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
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
        description="Plot DLA-completeness and unreachability vs K")
    p.add_argument("--dim", type=int, help="Dimension to filter")
    p.add_argument("--threshold", type=float, default=0.99,
                   help="Overlap threshold for 'reachable' (default 0.99)")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def _blank(v):
    return v is None or v == ""


def wilson(k, n, z=1.96):
    """Return (p_hat, lower, upper) Wilson score interval. NaN if n==0."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z/denom) * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, center-half), min(1.0, center+half)


def scan_available(results_dir):
    combos = {}
    rp = Path(results_dir)
    if not rp.exists():
        return combos
    for f in rp.glob("*.csv"):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
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
        ks = ", ".join(f"K={k}:{n}" for k, n in sorted(combos[dim].items()))
        print(f"  --dim {dim}    [{ks}]")


def load_data(results_dir, dim, threshold):
    """Per K, split by check_dla. Returns dict K -> stats."""
    agg = {}
    rp = Path(results_dir)
    if not rp.exists():
        return agg
    for f in rp.glob("*.csv"):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                if int(row["dim"]) != dim:
                    continue
                K = int(row["K"])
                d = agg.setdefault(K, {
                    "accepts": 0, "rejected": 0,        # DLA fraction (check_dla rows)
                    "dla_ok": 0, "dla_tot": 0,          # reachable count among full-DLA targets
                    "no_ok": 0, "no_tot": 0,            # reachable count among no-DLA targets
                })
                checked = (str(row.get("check_dla", "")) == "1")
                timed_out = (str(row.get("timed_out", "")) == "1")
                nr = row.get("n_rejected", "")
                if not _blank(nr):
                    d["rejected"] += int(float(nr))
                    if not timed_out:
                        d["accepts"] += 1
                ov = row.get("overlap", "")
                if not _blank(ov):
                    ok = 1 if float(ov) >= threshold else 0
                    if checked:
                        d["dla_ok"] += ok; d["dla_tot"] += 1
                    else:
                        d["no_ok"] += ok; d["no_tot"] += 1
    return agg


def plot_data(agg, dim, threshold):
    if not agg:
        print("No data matching dim.")
        return None
    Ks = np.array(sorted(agg.keys()), dtype=float)

    # series 1: DLA complete fraction
    dla_c = [wilson(agg[K]["accepts"], agg[K]["accepts"]+agg[K]["rejected"]) for K in Ks]
    # series 2: unreachability among full-DLA targets = 1 - reachable
    unr_dla = [wilson(agg[K]["dla_tot"]-agg[K]["dla_ok"], agg[K]["dla_tot"]) for K in Ks]
    # series 3: unreachability among no-DLA targets
    unr_no = [wilson(agg[K]["no_tot"]-agg[K]["no_ok"], agg[K]["no_tot"]) for K in Ks]

    def unpack(series):
        p = np.array([s[0] for s in series])
        lo = np.array([s[0]-s[1] for s in series])   # err below
        hi = np.array([s[2]-s[0] for s in series])   # err above
        lo = np.where(np.isfinite(lo), np.maximum(lo, 0.0), 0.0)
        hi = np.where(np.isfinite(hi), np.maximum(hi, 0.0), 0.0)
        return p, np.vstack([lo, hi])

    p1, e1 = unpack(dla_c)
    p2, e2 = unpack(unr_dla)
    p3, e3 = unpack(unr_no)

    dx = 0.12 * (Ks[1]-Ks[0]) if len(Ks) > 1 else 0.12
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(Ks, p1, yerr=e1, fmt="o", color="tab:blue",
                capsize=3, label="DLA complete")
    ax.errorbar(Ks-dx, p2, yerr=e2, fmt="s", color="tab:red",
                markerfacecolor="tab:red", capsize=3,
                label="unreachability prob. (full DLA)")
    ax.errorbar(Ks+dx, p3, yerr=e3, fmt="s", color="tab:red",
                markerfacecolor="none", capsize=3,
                label="unreachability prob.")

    ax.set_xlabel("K (number of generators)")
    ax.set_ylabel("fraction (DLA complete, unreachable states)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(Ks)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def main():
    setup_font()
    args = parse_args()
    script_dir = Path(__file__).parent.parent
    results_dir = script_dir / "spectral_canonical_results"

    if args.dim is None:
        show_available(scan_available(results_dir))
        return

    agg = load_data(results_dir, args.dim, args.threshold)
    fig = plot_data(agg, args.dim, args.threshold)
    if fig is None:
        return
    if args.out:
        fig.savefig(args.out, dpi=150); print(f"Saved to {args.out}")
    elif can_connect_wayland() or os.environ.get("DISPLAY"):
        plt.show()
    else:
        out = f"spectral_dla_dim{args.dim}.png"
        fig.savefig(out, dpi=150)
        print(f"Headless environment detected. Saved to {out}")


if __name__ == "__main__":
    main()
