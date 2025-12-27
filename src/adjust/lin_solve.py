import numpy as np

def solve_constrained(N: np.ndarray, u: np.ndarray, B: np.ndarray) -> np.ndarray:
    Z = np.zeros((6, 6), dtype=float)
    K = np.block([[N, B],
                  [B.T, Z]])
    rhs = np.concatenate([u, np.zeros(6, dtype=float)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(K) @ rhs
    return sol[:N.shape[0]]
