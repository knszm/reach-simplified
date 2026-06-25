import jax
import time
import dla
import numpy as np
from scipy.linalg import null_space
def select_random(key, H_k, K):
    n = H_k.shape[0]
    idx = jax.random.choice(key, n, shape=(K,), replace=False)
    return H_k[idx], idx

def generically_have_nontrivial_real_linear_eigendependence(H_k, Nsamples,initial_Nsamples=None, return_nullspace=False, disregard_sum=True):
    if initial_Nsamples is None:
        initial_Nsamples = Nsamples // 3
    K = H_k.shape[0]

    #sample initial_Nsamples weights, create Hamiltonians, with normal real iid weights.
    #for each calculate max_perturbation and operator norm
    #select the one weight with the largest max_perturbation/operator norm
    best_ratio = -1.0; best_lam = None
    for _ in range(initial_Nsamples):
        lam = np.random.normal(size=(K))
        H0 = np.tensordot(lam, H_k, axes=1)

        ratio = max_perturbation(H_k, lam) / np.linalg.norm(H0, ord=2)
        if ratio > best_ratio:
            best_ratio = ratio; best_lam = lam

  #  print(f'best: {best_ratio}')
    #... or return True, if the `largest` is still tiny.
    if best_ratio < 1e-3:
        return True
    #for this weight , and remaining Nsamples-initial_Nsamples, do the following:
    #sample weights in a ball with radius `max_perturbation`.
    radius = max_perturbation(H_k, best_lam)
 #   print(f'radius: {radius}')
    samples = []
    for _ in range(Nsamples - initial_Nsamples):
        d = np.random.normal(size= (K));
        d = d / np.linalg.norm(d)
        r = radius * np.random.uniform()**(1.0/K)
        lam = best_lam + r * d
    #generate hamiltonians
        samples.append(np.tensordot(lam, H_k, axes=1))
    #pass to linear_relations_between_eigenvalues
    ns = real_linear_relations_between_eigenvalues(samples)
    if return_nullspace:
        return ns
    return ns.shape[1] > (1 if disregard_sum else 0)

def real_linear_relations_between_eigenvalues(sampled_hamiltonians, tolerance=1e-9):
    eigenvalues = [np.linalg.eigvalsh(H).real for H in sampled_hamiltonians]
    return null_space(np.array(eigenvalues), rcond=tolerance)

def max_perturbation(H_k, lambdas0): #... that preserves eigenvalue order (no crossing)
#from https://en.wikipedia.org/wiki/Weyl%27s_inequality:
# difference of eigenvalues lambda_k-lambda'_k 
# under perturbed Hamiltonian H+V <= |V|.
# minimal shift that exchanges eigenvalues = minimal gap/2 (both move towards center)
# … so if |V|<minimal_gap/2, guaranteed no crossing
# and here we translate it into weights, because V=\sum epsilon_i H_i,
# and if \sum |H_i| |epsilon_i| < minimal_gap/2,
# then surely any |\sum H_i epsilon_i| < minimal_gap/2.
    H0=np.tensordot(lambdas0,H_k,axes=1)
    
    energies=np.linalg.eigvalsh(H0)
    min_gap =np.min(np.diff(energies))
    
    norms=np.array([np.linalg.norm(h, ord=2) for h in H_k])
    return min_gap/(2.0*np.sum(norms))


def select_random_full_dla(key, H_k, K, timeout=60.0, return_try_count=False, reject_function=None):
    start = time.time()
    n_rejected=0
    dim=H_k.shape[1]
    while time.time() - start < timeout:
        key, sub = jax.random.split(key)
        H_sub, idx = select_random(sub, H_k, K)
        if reject_function is None or not reject_function(idx, dim):
            if dla.full_dla(list(H_sub)):
                if return_try_count:
                    return (H_sub, idx), n_rejected
                else:
                    return (H_sub, idx)
        n_rejected+=1
    if return_try_count:
        return None, n_rejected
    else:
        return None
