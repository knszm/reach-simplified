import jax
import time
import dla

def select_random(key, H_k, K):
    n = H_k.shape[0]
    idx = jax.random.choice(key, n, shape=(K,), replace=False)
    return H_k[idx], idx



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
