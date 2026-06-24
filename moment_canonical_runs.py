import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--check_dla", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--n_states", type=int, default=5)
    return p.parse_args()

import jax, jax.numpy as jnp
import numpy as np
import canonical, hamiltonians, dla, moment

def make_keys(seed, N):
    master = jax.random.PRNGKey(seed)
    k_phi, k_ham, k_init = jax.random.split(master, 3)
    return (jax.random.split(k_phi, N),
            jax.random.split(k_ham, N),
            jax.random.split(k_init, N))

def one_run(run, dim, K, phi_key, ham_key, init_key,
            check_dla,  timeout,  n_states):
    psi = canonical.init_state(dim)
    phi = canonical.random_state(phi_key, dim)
    H_full = canonical.canonical_full(dim)

    if check_dla:
        reject_function=lambda idx, dim: not canonical.connected_spanning(idx, dim)
        out, n_rejected = hamiltonians.select_random_full_dla(
            ham_key, H_full, K, timeout, return_try_count=True, reject_function=reject_function)
        if out is None:                       # timed out
            return [dict(run=run, timed_out=1, n_rejected=n_rejected)]
        H_sub, idx = out
        dla_full = 1                          # found => full by construction
    else:
        H_sub, idx = hamiltonians.select_random(ham_key, H_full, K)
        dla_full = None                       # not checked => blank
        n_rejected=None


    phi_keys = jax.random.split(phi_key, n_states)
    rows = []
    for st in range(n_states):
        phi = canonical.random_state(phi_keys[st], dim)
        #inits = jax.random.normal(init_keys[st], (n_starts, K))
        margin=float(moment.moment_definiteness(H_sub, phi,psi))
        rows.append(dict(run=run, state=st, timed_out=0, dla_full=dla_full,
                         margin=margin,
                         n_rejected=(n_rejected if st == 0 else None),
                         indices=" ".join(map(str, idx))
                         ))
    return rows


import csv, os, uuid, datetime

HEADER = ["run","state","dim","K","seed","check_dla","dla_full","timed_out",
          "margin","n_rejected","indices","timestamp"]

def main():
    a = parse_args()
    phi_keys, ham_keys, init_keys = make_keys(a.seed, a.N)
    os.makedirs("moment_canonical_results", exist_ok=True)
    path = os.path.join("moment_canonical_results",
        f"dim{a.dim}_K{a.K}_N{a.N}_seed{a.seed}_{uuid.uuid4().hex[:8]}.csv")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, restval="")
        w.writeheader()
        for run in range(a.N):

            for r in one_run(run, a.dim, a.K, phi_keys[run], ham_keys[run],
                             init_keys[run], a.check_dla, 
                             a.timeout,  a.n_states):
                r.update(dim=a.dim, K=a.K, seed=a.seed,
                         check_dla=int(a.check_dla), 
                         timestamp=datetime.datetime.now().isoformat())
                w.writerow(r)
    print(path)

if __name__ == "__main__":
    main()
