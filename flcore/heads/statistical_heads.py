import numpy as np

class Bundle(dict):
    __getattr__ = dict.__getitem__

def derive_train(ptf, pty, ptc, C, K):
    ptf = ptf.astype(np.float64, copy=False)
    pty = pty.astype(np.int64, copy=False)
    ptc = ptc.astype(np.int64, copy=False)
    DIM = ptf.shape[1]
    ncount = np.zeros((K, C), dtype=np.int64)
    for k in range(K):
        yk = pty[ptc == k]
        ncount[k] = np.bincount(yk, minlength=C)
    globally_absent = np.flatnonzero(ncount.sum(0) == 0)
    if len(globally_absent):
        raise RuntimeError(f'Cannot compute global class means: classes {globally_absent.tolist()} have no training samples.')
    m_global = np.stack([ptf[pty == c].mean(0) for c in range(C)])
    Xc = ptf - m_global[pty]
    Sig = Xc.T @ Xc / len(Xc)
    lam = 0.1 * np.trace(Sig) / DIM
    m_grand = ptf.mean(0)
    Xt = ptf - m_grand
    Ncnt = ncount.sum(0).astype(float)
    pi_glob = Ncnt / Ncnt.sum()
    logpi = np.log(pi_glob + 1e-12)
    Sig_t_ex = Xt.T @ Xt / (len(Xt) - 1)
    Si_t_ex = np.linalg.inv(Sig_t_ex + 1e-06 * np.trace(Sig_t_ex) / DIM * np.eye(DIM))
    return Bundle(C=C, K=K, DIM=DIM, ptf=ptf, pty=pty, m_global=m_global, Sig=Sig, lam=lam, logpi=logpi, Si_t_ex=Si_t_ex)

def head_fedncm(S):
    M = S.m_global / np.linalg.norm(S.m_global, axis=1, keepdims=True)
    return [(M, np.zeros(S.C))] * S.K

def head_fedcgs_exactnorm(S):
    (Wte, bte) = affine_gaussian_head(S.m_global, S.Si_t_ex)
    return [(Wte, bte + S.logpi)] * S.K

def head_ccvr(S, m_per=200, head_seed=1):
    from sklearn.linear_model import LogisticRegression
    (C, DIM) = (S.C, S.DIM)
    rng = np.random.RandomState(head_seed)
    (Xs, ys) = ([], [])
    for c in range(C):
        fc = S.ptf[S.pty == c]
        mu = S.m_global[c]
        Sc = np.cov(fc.T) if len(fc) > 1 else S.Sig
        Sc = 0.5 * Sc + 0.5 * S.Sig + S.lam * np.eye(DIM)
        L = np.linalg.cholesky(Sc)
        z = rng.randn(m_per, DIM)
        Xs.append(mu + z @ L.T)
        ys.append(np.full(m_per, c))
    Xs = np.vstack(Xs)
    ys = np.concatenate(ys)
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xs, ys)
    cls = lr.classes_
    W = np.zeros((C, DIM))
    b = np.zeros(C)
    W[cls] = lr.coef_
    b[cls] = lr.intercept_
    return [(W, b)] * S.K

def affine_gaussian_head(M, Sinv):
    Wk = M @ Sinv
    return (Wk, -0.5 * np.einsum('cd,cd->c', Wk, M))
