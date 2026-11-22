"""3.2 VaR al 95% con redes GAN (500 escenarios por día).

Para cada uno de los N_TEST días de prueba se re-entrena la GAN con los 500 rendimientos
inmediatamente anteriores, se generan 500 escenarios y se calcula el VaR a 1 día.
El día k=0 (VaR al cierre de la ventana de entrenamiento) guarda además sus escenarios
y la curva de pérdidas para el análisis del notebook.
"""
import time
from multiprocessing import Pool

import pandas as pd
import torch

import config
import utils
from gan import ReturnGAN


def run_day(k: int):
    """Re-entrena la GAN para el día de prueba k y devuelve su VaR (corre en un proceso hijo)."""
    torch.set_num_threads(1)
    r = utils.load_returns().values
    t0 = time.time()
    model = ReturnGAN(n_assets=r.shape[1], seed=config.SEED + k, **config.GAN_PARAMS)
    model.fit(utils.train_window(r, k))
    scen = model.sample(config.N_SCENARIOS, seed=config.SEED + k)
    var, es = utils.var_es(utils.portfolio_return(scen))
    realized = utils.portfolio_return(r[config.N_TRAIN + k])[0]
    print(f"[GAN] día {k + 1:2d}/{config.N_TEST}  VaR={var:.4%}  "
          f"real={realized:+.4%}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(k=k, var_95=var, es_95=es, realized_return=realized, scen=scen,
                history=model.history)


def main():
    df = utils.load_returns()
    with Pool(min(config.N_TEST, 8)) as pool:       # un modelo independiente por día
        results = pool.map(run_day, range(config.N_TEST))

    rows = [dict(date=df.index[config.N_TRAIN + x["k"]], var_95=x["var_95"], es_95=x["es_95"],
                 realized_return=x["realized_return"],
                 hit=int(x["realized_return"] < -x["var_95"])) for x in results]
    first = results[0]                                # VaR al cierre de la ventana de entrenamiento
    pd.DataFrame(first["scen"], columns=df.columns).to_csv(
        config.RESULTS_DIR / "escenarios_gan_t0.csv", index=False)
    pd.DataFrame(first["history"], columns=["epoch", "loss_D", "loss_G"]).to_csv(
        config.RESULTS_DIR / "gan_perdidas_t0.csv", index=False)

    out = pd.DataFrame(rows).set_index("date")
    out.to_csv(config.RESULTS_DIR / "var_gan.csv")
    print(f"\nGuardado results/var_gan.csv  (excepciones: {out['hit'].sum()})")


if __name__ == "__main__":
    main()
