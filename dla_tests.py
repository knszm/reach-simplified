import argparse
import jax, datetime
import csv, os, uuid
import canonical, hamiltonians, dla


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def run_trials(dim, K, n_trials, seed):
    key = jax.random.PRNGKey(seed)
    Hf = canonical.canonical_full(dim)
    rows = []
    for trial in range(n_trials):
        key, sub = jax.random.split(key)
        Hs, idx = hamiltonians.select_random(sub, Hf, K)
        d = len(dla.dla_naive(list(Hs)))
        rows.append((dim, K, trial, int(d == dim**2-1), d,
                     ' '.join(map(str, idx)), datetime.datetime.now().isoformat()))
    return rows


HEADER = ["dim", "K", "trial", "is_full", "dla_dim", "indices", "timestamp"]


def save(rows, dim, K, seed, outdir="dla_results"):
    os.makedirs(outdir, exist_ok=True)
    short = uuid.uuid4().hex[:8]
    path = os.path.join(outdir, f"dim{dim}_K{K}_seed{seed}_{short}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


if __name__ == "__main__":
    a = parse_args()
    rows = run_trials(a.dim, a.K, a.n_trials, a.seed)
    print(save(rows, a.dim, a.K, a.seed))
