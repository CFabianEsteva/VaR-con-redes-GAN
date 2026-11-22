"""Prueba de robustez: sensibilidad del VaR de la GAN a la semilla de entrenamiento."""
import numpy as np
import torch

import config
import utils
from gan import ReturnGAN


def var_gan_seed(seed: int) -> float:
    """VaR 95% en t0 (500 escenarios) de una GAN entrenada con la semilla dada."""
    torch.set_num_threads(1)
    r = utils.load_returns().values
    model = ReturnGAN(n_assets=r.shape[1], seed=seed, **config.GAN_PARAMS)
    model.fit(utils.train_window(r, 0))
    scen = model.sample(config.N_SCENARIOS, seed=seed)
    return utils.var_es(utils.portfolio_return(scen))[0]
