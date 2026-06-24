import jax.numpy as jnp
import jax
import numpy as np
import itertools
import functools

pauli_lookup = {'_': jnp.eye(2, dtype=complex),
          'I': jnp.eye(2, dtype=complex),
          '1': jnp.eye(2, dtype=complex),
          ' ': jnp.eye(2, dtype=complex),
          'X': jnp.array([[0, 1], [1, 0]], dtype=complex),
          'Y': jnp.array([[0, -1j], [1j, 0]], dtype=complex),
          'Z': jnp.array([[1, 0], [0, -1]], dtype=complex)}

def pauli_string_to_operator(string):
    return functools.reduce(jnp.kron, [pauli_lookup[c] for c in string])

def pauli_full(nqubits, weight_max, weight_min=None, only_labels=False):
    if weight_min is None: #only fixed weight, no lowers
        weight_min=weight_max
    labels=[]
    for w in range(weight_min,weight_max+1): 
        for support in itertools.combinations(range(nqubits),w): #which indices with non-1?
            for pauli_assignment in itertools.product('XYZ',repeat=w): #what paulis at these indices?
                s=['_']*nqubits
                for pos,p in zip(support,pauli_assignment):
                    s[pos]=p
                labels.append(''.join(s))
    if only_labels:
        return labels
    else:
        return jnp.array([pauli_string_to_operator(s) for s in labels])

def random_state_qubits(key, nqubits):
    kr, ki = jax.random.split(key)
    dim=2**nqubits
    v = jax.random.normal(kr, (dim,)) + 1j * jax.random.normal(ki, (dim,))
    return v / jnp.linalg.norm(v)

def init_state_qubits(nqubits):
    dim=2**nqubits
    return jnp.zeros(dim, dtype=complex).at[0].set(1.0)

