# VaR de mercado con Redes GAN

## Equipo

- Esteva Gallegos Carlos Fabián
- León Gómez Erick
- Rodríguez Muñoz Salma Annette

## Descripción

Proyecto final de riesgo de mercado en el que se estima el Valor en Riesgo (VaR) al 95 % de un portafolio equiponderado de tres acciones: NVDA (tecnología), JPM (financiero) y XOM (energía). Se usan 500 días de historia y 500 escenarios por modelo, y se comparan dos enfoques: una red generativa adversaria (GAN) y la simulación Monte Carlo con distribución normal multivariada. Ambos se evalúan con un backtest de 20 días  y se elige un modelo finalista.

## Estructura del proyecto

```
.
├── data/                           # Datos de entrada
│   ├── precios.csv                 #   precios ajustados de NVDA, JPM y XOM
│   └── rendimientos.csv            #   rendimientos logarítmicos diarios
├── src/                            # Código fuente
│   ├── config.py                   #   parámetros globales (activos, ventanas, GAN)
│   ├── utils.py                    #   carga de datos, portafolio, VaR/ES y pruebas de backtest
│   ├── gan.py                      #   GAN propia (PyTorch)
│   ├── montecarlo.py               #   simulación Monte Carlo (normal multivariada)
│   ├── robustez.py                 #   sensibilidad del VaR de la GAN a la semilla
│   ├── plotstyle.py                #   estilo de gráficas compartido
│   ├── 01_descarga_datos.py        #   3.1 descarga de precios y rendimientos
│   ├── 02_var_redes_gan.py         #   3.2 VaR con la GAN propia
│   ├── 03_var_montecarlo.py        #   3.3 VaR vía Monte Carlo
│   ├── 04_backtest.py              #   3.4 backtest a 20 días
│   └── 05_evaluacion.py            #   3.5 comparación de modelos y finalista
├── notebooks/                      # Análisis y resultados
│   ├── 01_eda.ipynb                #   análisis exploratorio de datos
│   ├── 02_var.ipynb                #   VaR con GAN y Monte Carlo
│   ├── 03_backtest.ipynb           #   backtest
│   └── 04_evaluation.ipynb         #   evaluación y elección del finalista
├── results/                        # CSV, JSON y TXT generados (VaR, backtest, finalista)
├── figures/                        # Gráficas de los notebooks
└── requirements.txt                # Entorno principal
```

## Ejecución

```bash
pip install -r requirements.txt
cd src
python 01_descarga_datos.py
python 02_var_redes_gan.py
python 03_var_montecarlo.py
python 04_backtest.py
python 05_evaluacion.py
```
