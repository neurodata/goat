import numpy as np


def _check_init_input(P0, n):
    row_sum = np.sum(P0, axis=0)
    col_sum = np.sum(P0, axis=1)
    tol = 1e-3
    msg = None
    if P0.shape != (n, n):
        msg = "`P0` matrix must have shape m' x m', where m'=n-m"
    elif (
        (~np.isclose(row_sum, 1, atol=tol)).any()
        or (~np.isclose(col_sum, 1, atol=tol)).any()
        or (P0 < 0).any()
    ):
        msg = "`P0` matrix must be doubly stochastic"
    if msg is not None:
        raise ValueError(msg)


def _split_matrix(X, n):
    # definitions according to Seeded Graph Matching [2].
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _doubly_stochastic(P, tol=1e-3, max_iter=1000):
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if it % 100 == 0:  # only check every so often to speed up
            if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
                np.abs(P_eps.sum(axis=0) - 1) < tol
            ).all():
                # All column/row sums ~= 1 within threshold
                break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c
    return P_eps
