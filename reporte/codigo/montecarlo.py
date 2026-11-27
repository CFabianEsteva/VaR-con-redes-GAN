"""Simulación Monte Carlo paramétrica: rendimientos log ~ Normal multivariada."""
import numpy as np


def simulate_mc(train_returns: np.ndarray, n_scenarios: int, seed: int) -> np.ndarray:
    """Genera `n_scenarios` vectores de rendimientos log a 1 día.

    Se estiman la media y la covarianza muestrales de la ventana y se correlacionan
    normales independientes con la descomposición de Cholesky: r = mu + L z.
    """
    rng = np.random.default_rng(seed)
    mu = train_returns.mean(axis=0)
    cov = np.cov(train_returns, rowvar=False)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_scenarios, len(mu)))
    return mu + z @ chol.T
