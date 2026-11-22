"""Comprobaciones sobre ydata-synthetic 1.4.0 (`VanillaGAN`) que explican la variante "defecto".

Se ejecuta con el entorno `gan_lib` y NO entrena nada (segundos):
    conda run -n gan_lib python src/verifica_libreria.py
Escribe results/verificacion_libreria.json, que lee el notebook 05.
"""
import inspect
import json
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np

import config
import utils
from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.base import BaseGANModel
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN


def distinct_rows(model, X, batch_size, n_calls=200):
    """Filas distintas que `get_data_batch` entrega al llamarlo como lo hace `fit`."""
    seen = set()
    for _ in range(n_calls):       # fit() lo llama sin cambiar la semilla (seed=0 por defecto)
        seen.update(map(tuple, model.get_data_batch(X, batch_size).round(12)))
    return len(seen)


def main():
    X = utils.load_returns().iloc[: config.N_TRAIN].values
    out = {"filas_totales": len(X)}
    for bs in (128, 500):
        params = ModelParameters(batch_size=bs, lr=2e-4, betas=(0.5, 0.999), noise_dim=8, layers_dim=64)
        model = RegularSynthesizer(modelname="gan", model_parameters=params)
        out[f"filas_distintas_batch_{bs}"] = distinct_rows(model, X, bs)
    # Ruido: fit() entrena con tf.random.normal; sample() usa random.uniform
    src_fit = inspect.getsource(VanilllaGAN.fit)
    src_sample = inspect.getsource(BaseGANModel.sample)
    out["fit_usa_ruido_normal"] = "tf.random.normal" in src_fit
    out["sample_usa_ruido_uniforme"] = "random.uniform" in src_sample
    path = config.RESULTS_DIR / "verificacion_libreria.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
