import jax.numpy as jnp
import jax
import numpy as np

def canonical_full(dim, only_labels=False):
    mats = []
    labels=[]
    for i in range(dim):
        for j in range(i+1, dim):
            if only_labels:
                labels.append(('X',i,j))
                labels.append(('Y',i,j))
            else:
                x = np.zeros((dim, dim), complex); x[i, j] = 1;  x[j, i] = 1            
                y = np.zeros((dim, dim), complex); y[i, j] = -1j; y[j, i] = 1j
                mats.append(x); mats.append(y)
        if i < dim-1:
            if only_labels:
                labels.append(('D',i,None))
            else:
                d = -np.eye(dim, dtype=complex); d[i, i] = dim-1
                mats.append(d)
    if only_labels:
        return labels
    else:
        return jnp.array(np.stack(mats))

def connected_spanning(indices, dim): 
#find-union algo basically, returns True if connection graph spanned by X/Y interactions has one contigious part
#used for early non-full-DLA rejection.
    indices = np.asarray(indices)
    labels=canonical_full(dim,only_labels=True)
    parent = list(range(dim))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    touched = set()
    for k in indices:
        kind, i, j = labels[int(k)]
        if kind == 'D':
            continue
        touched.add(i); touched.add(j)
        parent[find(i)] = find(j)
    if len(touched) != dim:
        return False
    return len({find(v) for v in range(dim)}) == 1

def random_state(key, dim):
    kr, ki = jax.random.split(key)
    v = jax.random.normal(kr, (dim,)) + 1j * jax.random.normal(ki, (dim,))
    return v / jnp.linalg.norm(v)

def init_state(dim):
    return jnp.eye(dim, dtype=complex)[0]
