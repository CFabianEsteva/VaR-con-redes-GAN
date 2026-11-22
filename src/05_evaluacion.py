"""3.5 Comparación de modelos y elección del finalista.

Criterios (rango 1 = mejor; gana la menor suma de rangos):
  1. |excepciones - esperadas|          (calibración de la cobertura)
  2. pérdida cuantílica media           (precisión del cuantil 5%)
  3. VaR medio                          (eficiencia de capital, a igualdad de cobertura)
  4. distancia KS al historial en t0    (fidelidad de la distribución generada)
Desempate: menor pérdida cuantílica. Además se reporta el IC bootstrap del VaR en t0
para mostrar el error de muestreo de usar solo 500 escenarios.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils

CORE = ["GAN propia", "Monte Carlo"]


def main():
    bt = pd.read_csv(config.RESULTS_DIR / "backtest_resumen.csv", index_col=0).loc[CORE]
    r = utils.load_returns().values
    hist_port = utils.portfolio_return(utils.train_window(r, 0))
    scen = {"GAN propia": "escenarios_gan_t0.csv", "Monte Carlo": "escenarios_mc_t0.csv"}

    rows = {}
    for name, f in scen.items():
        s = pd.read_csv(config.RESULTS_DIR / f).values
        port = utils.portfolio_return(s)
        var0, es0 = utils.var_es(port)
        lo, hi = utils.bootstrap_var_ci(port)
        ks = stats.ks_2samp(port, hist_port)
        rows[name] = {
            "VaR_t0": var0, "VaR_t0_IC95_inf": lo, "VaR_t0_IC95_sup": hi, "ES_t0": es0,
            "KS_vs_historial": ks.statistic, "KS_p": ks.pvalue,
            "curtosis_exceso": stats.kurtosis(port), "asimetria": stats.skew(port),
        }
    dist = pd.DataFrame(rows).T
    hist_var, hist_es = utils.var_es(hist_port)

    table = bt.join(dist)
    table["desv_excepciones"] = (table["excepciones"] - table["esperadas"]).abs()
    criteria = ["desv_excepciones", "perdida_cuantilica", "VaR_medio", "KS_vs_historial"]
    ranks = table[criteria].rank(method="min")
    table["suma_rangos"] = ranks.sum(axis=1)
    table = table.sort_values(["suma_rangos", "perdida_cuantilica"])
    finalist = table.index[0]

    table.to_csv(config.RESULTS_DIR / "evaluacion_resumen.csv")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250,
                           "display.max_columns", 30):
        print(table.T)
    print(f"\nReferencia historial (500 días, t0): VaR={hist_var:.4%}  ES={hist_es:.4%}")
    print(f"\n>>> Modelo finalista: {finalist}")
    (config.RESULTS_DIR / "finalista.txt").write_text(finalist + "\n")


if __name__ == "__main__":
    main()
