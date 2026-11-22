"""3.2-bis VaR al 95% con la GAN de una librería (ydata-synthetic 1.4.0, `VanillaGAN`).

Verificación independiente de la GAN propia (`02_var_redes_gan_propio.py`): mismo esquema
(ventana móvil de 500 días, re-entrenamiento diario, 500 escenarios, VaR a 1 día).

IMPORTANTE - se ejecuta con el entorno `gan_lib` (Python 3.11 + TensorFlow), no con `venvpro`:
    conda run -n gan_lib python src/02_var_redes_gan_libreria.py [--variante defecto|corregida|ambas]
ydata-synthetic 1.4.0 no instala en Python >= 3.12 y la 2.0.1 es un paquete obsoleto vacío.

Dos variantes, porque la lectura del código de la librería revela dos comportamientos que
afectan a esta tarea (comprobados empíricamente, ver notebook 05):
  * defecto    : uso documentado (`fit` + `.sample()`), batch_size=128.
                 - `fit` llama siempre a `get_data_batch` con la misma semilla, así que con
                   batch_size=128 solo ve 128 de las 500 filas en TODO el entrenamiento.
                 - `.sample()` genera ruido uniforme, pero `fit` entrena con ruido normal.
  * corregida  : batch_size=500 (el lote cubre todas las filas) y muestreo con ruido normal,
                 que es la distribución con la que se entrenó el generador. Mismo número de
                 actualizaciones del generador que la variante por defecto (800 x 2 ~ 400 x 4).
Arquitectura y preprocesado son los de la librería (capas densas dim/2dim/4dim, MinMaxScaler);
solo se igualan los hiperparámetros comunes con la GAN propia (lr, betas, ruido de dim. 8).
"""
import argparse
import contextlib
import io
import os
import tempfile
import time
import warnings
from multiprocessing import get_context

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
import utils

VARIANTES = {
    "defecto": dict(batch_size=128, epochs=400, ruido="uniforme"),
    "corregida": dict(batch_size=500, epochs=800, ruido="normal"),
}
LAYERS_DIM = 64     # misma anchura base que la GAN propia (la librería usa dim, 2*dim, 4*dim)


def run_day(args):
    variante, k = args
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    cfg = VARIANTES[variante]
    gp = config.GAN_PARAMS
    r = utils.load_returns()
    train = r.iloc[k : k + config.N_TRAIN]
    t0 = time.time()
    tf.keras.utils.set_random_seed(config.SEED + k)
    params = ModelParameters(batch_size=cfg["batch_size"], lr=gp["lr_g"], betas=(gp["beta1"], 0.999),
                             noise_dim=gp["latent_dim"], layers_dim=LAYERS_DIM)
    model = RegularSynthesizer(modelname="gan", model_parameters=params)
    # `fit` escribe checkpoints en ./cache y pinta progreso: se aísla en un directorio temporal
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd(); os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model.fit(train, TrainParameters(epochs=cfg["epochs"], sample_interval=10**9),
                          num_cols=list(train.columns), cat_cols=[])
                if cfg["ruido"] == "uniforme":
                    scen = np.asarray(model.sample(config.N_SCENARIOS))[: config.N_SCENARIOS]
                else:
                    z = tf.random.normal((cfg["batch_size"], gp["latent_dim"]))
                    fake = model.generator(z, training=False).numpy()
                    scen = np.asarray(model.processor.inverse_transform(fake))[: config.N_SCENARIOS]
        finally:
            os.chdir(cwd)
    var, es = utils.var_es(utils.portfolio_return(scen))
    realized = utils.portfolio_return(r.values[config.N_TRAIN + k])[0]
    print(f"[LIB-{variante}] día {k + 1:2d}/{config.N_TEST}  VaR={var:+.4%}  "
          f"real={realized:+.4%}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(k=k, var_95=var, es_95=es, realized_return=realized, scen=scen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=[*VARIANTES, "ambas"], default="ambas")
    ap.add_argument("--procesos", type=int, default=8)
    a = ap.parse_args()
    df = utils.load_returns()
    for variante in (VARIANTES if a.variante == "ambas" else [a.variante]):
        with get_context("spawn").Pool(a.procesos) as pool:
            results = pool.map(run_day, [(variante, k) for k in range(config.N_TEST)])
        rows = [dict(date=df.index[config.N_TRAIN + x["k"]], var_95=x["var_95"], es_95=x["es_95"],
                     realized_return=x["realized_return"],
                     hit=int(x["realized_return"] < -x["var_95"])) for x in results]
        pd.DataFrame(results[0]["scen"], columns=df.columns).to_csv(
            config.RESULTS_DIR / f"escenarios_gan_lib_{variante}_t0.csv", index=False)
        out = pd.DataFrame(rows).set_index("date")
        out.to_csv(config.RESULTS_DIR / f"var_gan_lib_{variante}.csv")
        print(f"\nGuardado results/var_gan_lib_{variante}.csv  (excepciones: {out['hit'].sum()})\n")


if __name__ == "__main__":
    main()
