import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nqubits", type=int, required=True)
    p.add_argument("--weight_max", type=int, required=True)
    p.add_argument("--weight_min", type=int, default=None)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--check_dla", action="store_true")
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--n_states", type=int, default=5)
    p.add_argument("--eigendep_reject", action="store_true")
    return p.parse_args()

import jax, jax.numpy as jnp
import numpy as np
import paulis, hamiltonians, dla, spectral

def make_keys(seed, N):
    master = jax.random.PRNGKey(seed)
    k_phi, k_ham, k_init = jax.random.split(master, 3)
    return (jax.random.split(k_phi, N),
            jax.random.split(k_ham, N),
            jax.random.split(k_init, N))

def one_run(run, nqubits, weight_min,weight_max, K, phi_key, ham_key, init_key,
            check_dla, n_starts, timeout, threshold, n_states,eigendep_reject=False):
    psi = paulis.init_state_qubits(nqubits)
    H_full = paulis.pauli_full(nqubits, weight_max, weight_min)

    def reject_based_on_eigenvalue_lin_dep(idx,dim):
        if eigendep_reject == False:
            return False
        Hk=H_full[jnp.array(idx)]
        return hamiltonians.generically_have_nontrivial_real_linear_eigendependence(Hk,100)
        #...as a proxy of *rational linear eigendependence*


    if check_dla:
        out, n_rejected = hamiltonians.select_random_full_dla(
            ham_key, H_full, K, timeout, return_try_count=True,reject_function=reject_based_on_eigenvalue_lin_dep)
        if out is None:                       # timed out
            return [dict(run=run, timed_out=1, n_rejected=n_rejected)]
        H_sub, idx = out
        dla_full = 1                          # found => full by construction
    else:
        H_sub, idx = hamiltonians.select_random(ham_key, H_full, K)
        dla_full = None                       # not checked => blank
        n_rejected=None

    # multi-start

    phi_keys = jax.random.split(phi_key, n_states)
    init_keys = jax.random.split(init_key, n_states)
    rows = []
    for st in range(n_states):
        phi = paulis.random_state_qubits(phi_keys[st], nqubits)
        inits = jax.random.normal(init_keys[st], (n_starts, K))
        best_ov, best_lam = -1.0, None
        for s in range(n_starts):
            lam, ov = spectral.maximize_overlap(H_sub, psi, phi, inits[s])
            if ov > best_ov:
                best_ov, best_lam = ov, lam
            if best_ov > threshold:
                break
        rows.append(dict(run=run, state=st, timed_out=0, dla_full=dla_full,
                         overlap=best_ov,
                         n_rejected=(n_rejected if st == 0 else None),
                         indices=" ".join(map(str, idx)),
                         lam_opt=" ".join(map(str, np.asarray(best_lam)))))
    return rows

import csv, os, uuid, datetime

HEADER = ["run","state","nqubits","weight_min","weight_max","K","seed","check_dla","dla_full","timed_out",
          "overlap","n_rejected","indices","lam_opt","n_starts","timestamp","eigendep_reject"]

def main():
    a = parse_args()
    phi_keys, ham_keys, init_keys = make_keys(a.seed, a.N)
    os.makedirs("spectral_paulis_results", exist_ok=True)
    wmin=a.weight_min if a.weight_min is not None else a.weight_max # not provided => we sample only pauli strings of definite weight
    path = os.path.join("spectral_paulis_results",
        f"nqubits{a.nqubits}_w{wmin}-{a.weight_max}_K{a.K}_N{a.N}_seed{a.seed}_{uuid.uuid4().hex[:8]}.csv")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, restval="")
        w.writeheader()
        for run in range(a.N):
#            r = one_run(run, a.dim, a.K, phi_keys[run], ham_keys[run],
#                        init_keys[run], a.check_dla, a.n_starts, a.timeout, a.threshold)
#            r.update(dim=a.dim, K=a.K, seed=a.seed,
#                     check_dla=int(a.check_dla), n_starts=a.n_starts,
#                     timestamp=datetime.datetime.now().isoformat())
#            w.writerow(r)
            for r in one_run(run, a.nqubits,a.weight_min,a.weight_max, a.K, phi_keys[run], ham_keys[run],
                             init_keys[run], a.check_dla, a.n_starts,
                             a.timeout, a.threshold, a.n_states,a.eigendep_reject):
                r.update(nqubits=a.nqubits,weight_min=wmin,weight_max=a.weight_max, K=a.K, seed=a.seed,
                         check_dla=int(a.check_dla), n_starts=a.n_starts,
                         timestamp=datetime.datetime.now().isoformat(),eigendep_reject=a.eigendep_reject)
                w.writerow(r)
    print(path)

if __name__ == "__main__":
    main()
