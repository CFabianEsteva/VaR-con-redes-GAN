"""3.4 Backtest a 20 días: excepciones, Kupiec y pérdida cuantílica."""
import numpy as np
import pandas as pd

import config
import utils

MODELS = {
    "GAN propia": "var_gan.csv",
    "Monte Carlo": "var_montecarlo.csv",
}


def backtest(df: pd.DataFrame) -> dict:
    hits = df["hit"].to_numpy()
    n, x = len(df), int(hits.sum())
    lr_uc, p_uc = utils.kupiec_pof(hits)
    exceed = (-df["realized_return"] - df["var_95"])[hits == 1]
    return {
        "dias": n,
        "excepciones": x,
        "esperadas": n * config.ALPHA,
        "tasa_excepciones": x / n,
        "kupiec_LR": lr_uc, "kupiec_p": p_uc,
        "exceso_medio_sobre_VaR": exceed.mean() if x else 0.0,
        "VaR_medio": df["var_95"].mean(),
        "perdida_cuantilica": utils.pinball_loss(df["realized_return"], df["var_95"]),
    }


def main():
    res = {}
    for name, fname in MODELS.items():
        df = pd.read_csv(config.RESULTS_DIR / fname, index_col=0, parse_dates=True)
        res[name] = backtest(df)
    summary = pd.DataFrame(res).T
    summary.to_csv(config.RESULTS_DIR / "backtest_resumen.csv")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(summary)
    print("\nNota: con 20 días y VaR 95% se esperan ~1 excepción; los contrastes tienen "
          "muy poca potencia estadística.")


if __name__ == "__main__":
    main()
