"""Estilo de gráficas compartido por los notebooks.

Paleta categórica de referencia (slots 1-3, validados para todos los pares, incl. CVD):
activos NVDA / JPM / XOM = azul / naranja / aqua; modelos GAN / Monte Carlo = azul / naranja.
El texto va siempre en tinta neutra, nunca en el color de la serie.
"""
import matplotlib.pyplot as plt

import config

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ASSET_COLORS = dict(zip(config.TICKERS, [BLUE, ORANGE, AQUA]))
C_GAN, C_MC, C_HIST = BLUE, ORANGE, "#8a8984"
C_ALERT = "#e34948"          # excepciones del backtest (con marcador distinto al de la serie)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def apply():
    plt.rcParams.update({
        "figure.figsize": (9, 4), "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
        "font.size": 10,
    })


def save(fig, name: str):
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
