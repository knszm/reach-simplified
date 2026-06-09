"""Load experiment CSVs and reconstruct Hamiltonians/states."""
""" LLM-generated. """

import csv
import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm
import canonical


def load_runs(path):
    """Load CSV file. Returns list of dicts with parsed fields."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(_parse_row(row))
    return rows


def _parse_row(row):
    """Parse string fields into proper types."""
    out = dict(row)
    for k in ("dim", "K", "seed", "run", "state", "n_starts"):
        if k in out and out[k] != "":
            out[k] = int(out[k])
    for k in ("overlap",):
        if k in out and out[k] != "":
            out[k] = float(out[k])
    for k in ("check_dla", "dla_full", "timed_out"):
        if k in out and out[k] != "":
            out[k] = int(out[k]) == 1
    if out.get("indices", ""):
        out["indices"] = [int(x) for x in out["indices"].split()]
    else:
        out["indices"] = []
    if out.get("lam_opt", ""):
        out["lam_opt"] = np.array([float(x) for x in out["lam_opt"].split()])
    else:
        out["lam_opt"] = None
    return out


def get_H_sub(row, H_full=None):
    """Get the K selected Hamiltonian generators for this run."""
    if H_full is None:
        H_full = canonical.canonical_full(row["dim"])
    idx = row["indices"]
    return H_full[jnp.array(idx)]


def get_opt_hamiltonian(row, H_full=None):
    """Reconstruct the optimized Hamiltonian H = sum(lam[i] * H_sub[i])."""
    H_sub = get_H_sub(row, H_full)
    lam = row["lam_opt"]
    if lam is None:
        return None
    return jnp.tensordot(jnp.array(lam), H_sub, axes=1)


def get_opt_state(row, H_full=None, psi=None):
    """Reconstruct evolved state: exp(i*H_opt) @ psi."""
    H = get_opt_hamiltonian(row, H_full)
    if H is None:
        return None
    if psi is None:
        psi = canonical.init_state(row["dim"])
    H_np = np.array(H)
    U = expm(1j * H_np)
    return jnp.array(U @ np.array(psi))
