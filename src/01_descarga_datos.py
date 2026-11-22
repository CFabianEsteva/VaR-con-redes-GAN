"""3.1 Descarga de precios (NVDA, JPM, XOM) y cálculo de rendimientos logarítmicos.

Se guardan N_TRAIN + N_TEST + 1 precios ajustados: 500 días de entrenamiento más los
20 días del backtest (que quedan fuera de la muestra de entrenamiento).
"""
import numpy as np
import yfinance as yf

import config


def main():
    px = yf.download(config.TICKERS, start=config.DOWNLOAD_START, end=config.END_DATE,
                     auto_adjust=True, progress=False)["Close"][config.TICKERS]
    px = px.dropna()
    n_needed = config.N_TRAIN + config.N_TEST + 1
    if len(px) < n_needed:
        raise RuntimeError(f"Solo hay {len(px)} precios; se necesitan {n_needed}")
    px = px.iloc[-n_needed:]
    rets = np.log(px / px.shift(1)).dropna()

    px.to_csv(config.PRICES_CSV)
    rets.to_csv(config.RETURNS_CSV)
    print(f"Precios:       {px.index[0].date()} -> {px.index[-1].date()} ({len(px)} filas)")
    print(f"Rendimientos:  {len(rets)} filas "
          f"({len(rets) - config.N_TEST} entrenamiento + {config.N_TEST} prueba)")
    print(f"Guardado en {config.DATA_DIR}")


if __name__ == "__main__":
    main()
