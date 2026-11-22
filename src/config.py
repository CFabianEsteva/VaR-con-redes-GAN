"""Parámetros globales del proyecto: VaR de mercado con redes GAN."""
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(exist_ok=True)

PRICES_CSV = DATA_DIR / "precios.csv"
RETURNS_CSV = DATA_DIR / "rendimientos.csv"

# --- Activos y portafolio --------------------------------------------------
TICKERS = ["NVDA", "JPM", "XOM"]
WEIGHTS = [1 / 3, 1 / 3, 1 / 3]          # portafolio equiponderado, rebalanceo diario

# --- Ventanas --------------------------------------------------------------
N_TRAIN = 500        # días de historia para estimar cada modelo
N_TEST = 20          # días del backtest
# yfinance trata `end` como exclusivo: con "2026-11-18" el último dato es el
# 17-nov-2026 (cierre completo). Fijarlo hace el análisis reproducible.
END_DATE = "2026-11-18"
DOWNLOAD_START = "2023-06-01"   # margen de sobra para obtener 521 precios

# --- Riesgo ----------------------------------------------------------------
CONFIDENCE = 0.95
ALPHA = 1 - CONFIDENCE
N_SCENARIOS = 500    # escenarios por modelo (según el enunciado)

# --- GAN -------------------------------------------------------------------
SEED = 42
GAN_PARAMS = dict(
    window=1,          # días por muestra (1 = vector diario de 3 activos)
    latent_dim=8,
    hidden=64,
    epochs=400,
    batch_size=128,
    lr_g=2e-4,
    lr_d=2e-4,
    beta1=0.5,
    ema=0.99,          # media móvil de pesos del generador (estabiliza el muestreo)
)
