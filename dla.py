import numpy as np

def comm(A, B):
    return (A @ B - B @ A) / 1j

def _vec(M):
    f = np.array(M, complex).reshape(-1)
    return np.concatenate([f.real, f.imag])

def _dla_naive(gens, tol=1e-9):
    basis = [np.array(g, complex) for g in gens]
    added = True
    while added:
        added = False
        m = len(basis)                 # snapshot: freeze both bounds
        for a in range(m):
            for b in range(a+1, m):
                vecs = [_vec(x) for x in basis]
                c = comm(basis[a], basis[b])
                r0 = np.linalg.matrix_rank(np.stack(vecs), tol=tol)
                r1 = np.linalg.matrix_rank(np.stack(vecs + [_vec(c)]), tol=tol)
                if r1 > r0:
                    basis.append(c); added = True
    return basis



def _dla_optimized(gens, tol=1e-9):
    dim = gens[0].shape[0]
    cap = dim**2 - 1
    basis = [np.array(g, complex) for g in gens]
    vecs = [_vec(x) for x in basis]
    r0 = np.linalg.matrix_rank(np.stack(vecs), tol=tol)
    added = True
    while added:
        added = False
        m = len(basis)
        for a in range(m):
            for b in range(a+1, m):
                c = comm(basis[a], basis[b])
                cv = _vec(c)
                r1 = np.linalg.matrix_rank(np.stack(vecs + [cv]), tol=tol)
                if r1 > r0:
                    basis.append(c); vecs.append(cv)
                    r0 = r1
                    added = True
                    if r0 == cap:
                        return basis
    return basis
def _orthonormalize(Q, v, tol):
    w = v.copy()
    for q in Q:
        w = w - (q @ w) * q
    nrm = np.linalg.norm(w)
    if nrm > tol:
        return w / nrm
    return None

def _dla_qr(gens, tol=1e-9):
    dim = gens[0].shape[0]
    cap = dim**2 - 1
    basis = []
    Q = []
    for g in gens:
        g = np.array(g, complex)
        u = _orthonormalize(Q, _vec(g), tol)
        basis.append(g)
        if u is not None:
            Q.append(u)
    added = True
    while added:
        added = False
        m = len(basis)
        for a in range(m):
            for b in range(a+1, m):
                c = comm(basis[a], basis[b])
                u = _orthonormalize(Q, _vec(c), tol)
                if u is not None:
                    basis.append(c); Q.append(u)
                    added = True
                    if len(Q) == cap:
                        return basis
    return basis


def _dla_qr_faster(gens, tol=1e-9): ##LLM-generated, checked and yields same dim as _dla_qr
    dim = gens[0].shape[0]; cap = dim**2 - 1
    basis = []; Q = []
    for g in gens:
        g = np.array(g, complex)
        u = _orthonormalize(Q, _vec(g), tol)
        basis.append(g)
        if u is not None:
            Q.append(u)
    old_count = 0
    frontier = list(range(len(basis)))
    while frontier:
        new_idx = []
        cur = len(basis)
        for i in frontier:
            for j in range(i):
                if j < old_count and i < old_count:
                    continue                      # both old -> skip
                c = comm(basis[j], basis[i])
                u = _orthonormalize(Q, _vec(c), tol)
                if u is not None:
                    basis.append(c); Q.append(u); new_idx.append(len(basis)-1)
                    if len(Q) == cap:
                        return basis
        old_count = cur
        frontier = new_idx
    return basis

def _dla_dim_via_rank(basis, tol=1e-9):
    V = np.stack([_vec(np.array(m, complex)) for m in basis])
    return np.linalg.matrix_rank(V, tol=tol)

def dla(gens,tol=1e-9):
    return _dla_optimized(gens,tol)

def full_dla(gens, tol=1e-9):
    assert len(gens)>0
    #dim = gens[0].shape[0] 
    #dla_dim = len(_dla_optimized(gens,tol))
    dim = gens[0].shape[0]
    basis = _dla_qr_faster(gens, tol)
    dla_dim = _dla_dim_via_rank(basis, tol) 
    assert dla_dim <= dim**2 -1
    return dla_dim == dim**2 -1
