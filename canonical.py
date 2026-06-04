import jax.numpy as jnp
import jax
import numpy as np

def canonical_full(dim):
    mats = []
    for i in range(dim):
        for j in range(i+1, dim):
            x = np.zeros((dim, dim), complex); x[i, j] = 1;  x[j, i] = 1
            y = np.zeros((dim, dim), complex); y[i, j] = -1j; y[j, i] = 1j
            mats.append(x); mats.append(y)
        if i < dim-1:
            d = -np.eye(dim, dtype=complex); d[i, i] = dim-1
            mats.append(d) 
    return jnp.array(np.stack(mats))

def random_state(key, dim):
    kr, ki = jax.random.split(key)
    v = jax.random.normal(kr, (dim,)) + 1j * jax.random.normal(ki, (dim,))
    return v / jnp.linalg.norm(v)

def init_state(dim):
    return jnp.eye(dim, dtype=complex)[0]
