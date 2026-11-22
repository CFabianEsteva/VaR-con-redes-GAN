"""3.3 VaR al 95% vía Monte Carlo (500 escenarios por día, normal multivariada).

Mismo esquema que la GAN: ventana móvil de 500 días, 500 escenarios, VaR a 1 día.
"""
import pandas as pd

import config
import utils
from montecarlo import simulate_mc


def main():
    df = utils.load_returns()
    r = df.values
    rows = []
    for k in range(config.N_TEST):
        train = utils.train_window(r, k)
        scen = simulate_mc(train, config.N_SCENARIOS, seed=config.SEED + k)
        port = utils.portfolio_return(scen)
        var, es = utils.var_es(port)
        realized = utils.portfolio_return(r[config.N_TRAIN + k])[0]
        rows.append(dict(date=df.index[config.N_TRAIN + k], var_95=var, es_95=es,
                         realized_return=realized, hit=int(realized < -var)))
        print(f"[MC ] día {k + 1:2d}/{config.N_TEST}  VaR={var:.4%}  real={realized:+.4%}")
        if k == 0:
            pd.DataFrame(scen, columns=df.columns).to_csv(
                config.RESULTS_DIR / "escenarios_mc_t0.csv", index=False)

    out = pd.DataFrame(rows).set_index("date")
    out.to_csv(config.RESULTS_DIR / "var_montecarlo.csv")
    print(f"\nGuardado results/var_montecarlo.csv  (excepciones: {out['hit'].sum()})")


if __name__ == "__main__":
    main()
