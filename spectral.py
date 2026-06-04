import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

def overlap(lambdas, H_k, psi, phi, eps=1e-10):
    H = jnp.tensordot(lambdas, H_k, axes=1)
    evals, evecs = jnp.linalg.eigh(H)
    psi = psi / jnp.linalg.norm(psi)
    phi = phi / jnp.linalg.norm(phi)
    a = evecs.conj().T @ psi
    b = evecs.conj().T @ phi
    p = jnp.abs(a)**2
    q = jnp.abs(b)**2
    return jnp.sum(jnp.sqrt(p * q + eps))

def maximize_overlap(H_k, psi, phi, lambdas0, method="L-BFGS-B", _alpha=1.0, _sigma=1.0):
    def _penalty(lam):
        return _alpha * jnp.exp(-jnp.sum(lam**2) / _sigma**2)

    def obj(lam):
        return -overlap(lam, H_k, psi, phi) + _penalty(lam)

    g = jax.grad(obj)
    def f_np(x):
        return float(obj(jnp.asarray(x)))
    def g_np(x):
        return np.asarray(g(jnp.asarray(x)), dtype=float)
    x0 = np.asarray(lambdas0)
    res = None
    if np.all(np.isfinite(g_np(x0))):
        res = minimize(f_np, x0, jac=g_np, method=method)
    if res is None or not np.isfinite(res.fun) or getattr(res,"nit",0)<=2:
        res = minimize(f_np, x0, method="Powell",
                       options={"maxiter": 20000, "maxfev": 20000})
    lam_opt = jnp.asarray(res.x)
    return lam_opt, float(overlap(lam_opt, H_k, psi, phi,eps=0.0))
