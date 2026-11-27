"""Genera las tablas de LaTeX del reporte a partir de results/*.csv.

Uso (desde la carpeta reporte/):  python3 generar_tablas.py
Solo usa la biblioteca estándar, para no depender del entorno de los modelos.
"""
import csv
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RESULTS = AQUI.parent / "results"
OUT = AQUI / "tablas"
OUT.mkdir(exist_ok=True)

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def leer(nombre):
    with open(RESULTS / nombre, newline="") as f:
        return list(csv.DictReader(f))


def fecha(s):
    y, m, d = s.split("-")
    return f"{int(d)}-{MESES[int(m) - 1]}"


def pct(x, dec=2, signo=False):
    v = float(x) * 100
    txt = f"{v:+.{dec}f}" if signo else f"{v:.{dec}f}"
    return txt.replace("-", "$-$") + r"\,\%"


def main():
    gan, mc = leer("var_gan.csv"), leer("var_montecarlo.csv")
    assert len(gan) == len(mc) == 20

    # --- Tabla diaria del backtest -------------------------------------------------
    filas = []
    for k, (g, m) in enumerate(zip(gan, mc), start=1):
        assert g["date"] == m["date"] and g["realized_return"] == m["realized_return"]
        def celdas(r):
            # Las excepciones se marcan en negrita
            if r["hit"] == "1":
                return f"\\textbf{{{pct(r['var_95'])}}} & \\textbf{{Sí}}"
            return f"{pct(r['var_95'])} & --"
        filas.append(
            f"{k} & {fecha(g['date'])} & {pct(g['realized_return'], signo=True)} & "
            f"{celdas(g)} & {celdas(m)} \\\\"
        )
    cuerpo = "\n".join(filas)
    (OUT / "backtest_diario.tex").write_text(
        "\\begin{tabular}{@{}rlrrcrc@{}}\n\\toprule\n"
        " & & & \\multicolumn{2}{c}{\\textbf{GAN propia}} & \\multicolumn{2}{c}{\\textbf{Monte Carlo}} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "\\textbf{Día} & \\textbf{Fecha} & \\textbf{Rend. real} & \\textbf{VaR 95\\,\\%} & \\textbf{Exc.} "
        "& \\textbf{VaR 95\\,\\%} & \\textbf{Exc.} \\\\\n\\midrule\n"
        f"{cuerpo}\n\\bottomrule\n\\end{{tabular}}\n"
    )

    # --- Estadísticos del VaR a lo largo de los 20 días ---------------------------
    def stats(rows, col):
        v = sorted(float(r[col]) for r in rows)
        n = len(v)
        media = sum(v) / n
        sd = (sum((x - media) ** 2 for x in v) / (n - 1)) ** 0.5
        return media, sd, v[0], v[-1]

    lineas = []
    for nombre, rows in (("GAN propia", gan), ("Monte Carlo", mc)):
        for col, etiqueta in (("var_95", "VaR 95\\,\\%"), ("es_95", "ES 95\\,\\%")):
            mu, sd, lo, hi = stats(rows, col)
            lineas.append(f"{nombre} & {etiqueta} & {pct(mu)} & {pct(sd, 2)} & {pct(lo)} & {pct(hi)} \\\\")
    (OUT / "var_20dias.tex").write_text(
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
        "\\textbf{Modelo} & \\textbf{Medida} & \\textbf{Media} & \\textbf{Desv.} & \\textbf{Mín.} & \\textbf{Máx.} \\\\\n"
        "\\midrule\n" + "\n".join(lineas) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    print("Tablas generadas en", OUT)


if __name__ == "__main__":
    main()
