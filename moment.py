import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

def anticommutators_and_expvals(H_k, state):
    state=state/jnp.linalg.norm(state)
    #calculate list of states: (list) (H_1 |state>,...,H_K|state>)
    Hpsi = jnp.einsum('kab,b->ka',H_k,state)
    #from this, we can efficiently calculate both L and Q
    L=jnp.einsum('a,ka->k',jnp.conj(state),Hpsi)
    # calculate matrix Q:= <state|H_iH_j|state> 
    # ... as (list)^\dagger list:
    Q = jnp.einsum('ia,ja->ij',jnp.conj(Hpsi),Hpsi)
    # Q <- (Q+Q^\dagger)/2
    Q=(Q+jnp.conj(Q).T)/2
    return Q,L

def soft_min(e): return -jax.scipy.special.logsumexp(-beta * e) / beta
def soft_max(e): return  jax.scipy.special.logsumexp( beta * e) / beta

    
def moment_definiteness(H_k, psi, phi, etas=[-1000,1000]):
# returns min eigenvalue of Q+eta L² across etas... or min eigenvalue of -(Q+eta L²) if it's better (both show the same)
    Qpsi,Lpsi = anticommutators_and_expvals(H_k,psi)
    Qphi,Lphi = anticommutators_and_expvals(H_k,phi)
    deltaQ=Qpsi-Qphi
    deltaL=Lpsi-Lphi
    deltaLouter = jnp.outer(deltaL,deltaL)

    max_min_eigval=-1000000 #whatever below zero works. detects only if >0 at the end
    min_max_eigval=+1000000 #likewise
    for eta in etas:
        #calculate eigenvalues of deltaQ+eta deltaLouter
        eig = jnp.linalg.eigvalsh((deltaQ + eta * deltaLouter).real)
        #max_min_eigval=max(max_min_eigval,minimial eigenvalue)
        lo, hi = eig[0], eig[-1]          # eigvalsh ascending
        max_min_eigval = jnp.maximum(max_min_eigval, lo)
        min_max_eigval = jnp.minimum(min_max_eigval, hi)
        #min max eigval likewise
    return max(max_min_eigval,-min_max_eigval)
    #... or smooth surrogate

