import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.0)        # stddev of true lambdas
    return p.parse_args()

import jax, jax.numpy as jnp
import numpy as np
import canonical, hamiltonians, spectral

def make_keys(seed, N):
    master = jax.random.PRNGKey(seed)
    k_ham, k_lam, k_init = jax.random.split(master, 3)
    return (jax.random.split(k_ham, N),
            jax.random.split(k_lam, N),
            jax.random.split(k_init, N))

def one_run(run, dim, K, ham_key, lam_key, init_key, n_starts, scale):
    psi = canonical.init_state(dim)
    H_full = canonical.canonical_full(dim)
    H_sub, idx = hamiltonians.select_random(ham_key, H_full, K)

    # construct an explicitly reachable target: phi = exp(i H(lam_true)) psi
    lam_true = scale * jax.random.normal(lam_key, (K,))
    H = jnp.tensordot(lam_true, H_sub, axes=1)
    phi = jax.scipy.linalg.expm(1j * H) @ psi
    ov_true = float(spectral.overlap(lam_true, H_sub, psi, phi))

    # multi-start search
    inits = jax.random.normal(init_key, (n_starts, K))
    best_ov, best_lam = -1.0, None
    for s in range(n_starts):
        lam, ov = spectral.maximize_overlap(H_sub, psi, phi, inits[s])
        if ov > best_ov:
            best_ov, best_lam = ov, lam

    return dict(run=run, ov_true=ov_true, found_overlap=best_ov,
                indices=" ".join(map(str, idx)),
                true_lambdas=" ".join(map(str, np.asarray(lam_true))),
                lam_opt=" ".join(map(str, np.asarray(best_lam))))

import csv, os, uuid, datetime

HEADER = ["run","dim","K","seed","scale","ov_true",
          "found_overlap","indices","true_lambdas","lam_opt",
          "n_starts","timestamp"]

def main():
    a = parse_args()
    ham_keys, lam_keys, init_keys = make_keys(a.seed, a.N)
    os.makedirs("explicit_reachability_results", exist_ok=True)
    path = os.path.join("explicit_reachability_results",
        f"dim{a.dim}_K{a.K}_N{a.N}_seed{a.seed}_{uuid.uuid4().hex[:8]}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, restval="")
        w.writeheader()
        for run in range(a.N):
            r = one_run(run, a.dim, a.K, ham_keys[run], lam_keys[run],
                        init_keys[run], a.n_starts, a.scale)
            r.update(dim=a.dim, K=a.K, seed=a.seed, scale=a.scale,
                     n_starts=a.n_starts,
                     timestamp=datetime.datetime.now().isoformat())
            w.writerow(r)
    print(path)

if __name__ == "__main__":
    main()
