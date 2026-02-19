# What if ???

Esta seccion esta dedicada a explorar ciertos cambios que se hicieron en el dataset y los modelos, esto con el fin de ver como reacciona,
mas que nada por mera curiosidad y ver si en base a eso el modelo empeora o mejora.


Abarcaremos brevemente los siguiente escenarios:

#Seccion de What if?

1 - Qué pasaria si vieramos la distribución de los datos para encontrar alguna asimetria y corregirla, mejoraria el modelo?

2 - Qué tal si en vez de usar una seleccion de variables por contexto, buscamos la mejor combinacion de las variables para tratar de mejorarlo?

3 - Si creamos alguna interacción entre las variables, mejoraria el modelo?

4- si normalizamos los datos, el modelo mejoraria?


# 1) Ver distribucion

>PythoCode


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor("#0b0e1a")

vars_plot = ["pl_eqt", "pl_insol", "st_teff", "st_rad"]
colors    = ["#4fd1c5", "#a78bfa", "#f6ad55", "#68d391"]

for ax, col, color in zip(axes, vars_plot, colors):
    ax.set_facecolor("#111628")
    data = df_model_clean[col].dropna()

    ax.hist(data, bins=60, color=color, alpha=0.85, edgecolor="none")

    # Media y mediana
    ax.axvline(data.mean(),   color="#fc8181", linewidth=1.5, linestyle="--", label=f"Media: {data.mean():.1f}")
    ax.axvline(data.median(), color="#fff",    linewidth=1.5, linestyle=":",  label=f"Mediana: {data.median():.1f}")

    ax.set_title(col, color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors="#a0aec0", labelsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", color="#1e2540", linewidth=0.6)
    ax.legend(framealpha=0.2, labelcolor="#e2e8f0", fontsize=7.5, facecolor="#111628")

    # Skewness
    skew = data.skew()
    ax.text(0.97, 0.92, f"skew: {skew:.2f}", transform=ax.transAxes,
            color="#e2e8f0", fontsize=8, ha="right", fontfamily="monospace",
            bbox=dict(facecolor="#1e2540", alpha=0.6, edgecolor="none", pad=4))

fig.suptitle("Distribución de variables — df_model", color="#e2e8f0",
             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("distribucion_variables.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
```


>Output



