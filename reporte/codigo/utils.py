"""Utilidades comunes: carga de datos, portafolio, VaR/ES y pruebas de backtest."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

import config


# --- Datos -----------------------------------------------------------------
def load_returns() -> pd.DataFrame:
    """Rendimientos logarítmicos diarios (N_TRAIN + N_TEST filas x 3 activos)."""
    return pd.read_csv(config.RETURNS_CSV, index_col=0, parse_dates=True)


def load_prices() -> pd.DataFrame:
    return pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)


def train_window(returns: np.ndarray, k: int) -> np.ndarray:
    """Los N_TRAIN rendimientos que preceden al día de prueba k (0..N_TEST-1)."""
    return returns[k : k + config.N_TRAIN]


# --- Portafolio y riesgo ---------------------------------------------------
def portfolio_return(log_returns: np.ndarray, weights=None) -> np.ndarray:
    """Rendimiento simple del portafolio a partir de rendimientos log de cada activo."""
    w = np.asarray(config.WEIGHTS if weights is None else weights)
    return (np.exp(np.atleast_2d(log_returns)) - 1.0) @ w


def var_es(port_returns: np.ndarray, alpha: float = config.ALPHA):
    """VaR y ES a 1 día, expresados como pérdida positiva (fracción del portafolio)."""
    q = np.quantile(port_returns, alpha)
    tail = port_returns[port_returns <= q]
    return -q, -tail.mean()


def bootstrap_var_ci(port_returns: np.ndarray, alpha: float = config.ALPHA,
                     n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """IC por bootstrap del VaR estimado con un número finito de escenarios."""
    rng = np.random.default_rng(seed)
    n = len(port_returns)
    boots = np.array([
        -np.quantile(port_returns[rng.integers(0, n, n)], alpha) for _ in range(n_boot)
    ])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return lo, hi


# --- Pruebas de backtest ---------------------------------------------------
def kupiec_pof(hits: np.ndarray, alpha: float = config.ALPHA):
    """Kupiec (1995): contraste de cobertura incondicional. Devuelve (LR, p-valor)."""
    hits = np.asarray(hits, dtype=int)
    n, x = len(hits), hits.sum()
    pi_hat = x / n
    ll0 = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll1 = xlogy(n - x, 1 - pi_hat) + xlogy(x, pi_hat)
    lr = -2 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, df=1)


def pinball_loss(realized_returns: np.ndarray, var: np.ndarray, alpha: float = config.ALPHA):
    """Pérdida cuantílica media (menor = mejor). El cuantil predicho es -VaR."""
    q = -np.asarray(var)
    u = np.asarray(realized_returns) - q
    return np.mean(np.where(u >= 0, alpha * u, (alpha - 1) * u))
