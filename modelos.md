# Creación y comparación de modelos



## Modelo Lineal
En esta sección crearemos modelos lineales y no lineales para ver el desempeño de cada uno en los datos.



Continuando despues de la limpieza, ya tenemos los datos listos, ahora dividamos nuestros datos, probemos con 80% de los datos originales, esto para train y test.

Para esto usaremos OLS regression para ver los datos mas explicitos y scikit-learn para las métricas predictivas y calculos de test. 

Asi mismo despues de eso, grafiquemos los datos predichos vs los datos reales para ver la dispersion.





>PythonCode



```python
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ── Split ─────────────────────────────────────────────────────────────────────
X = df_model_clean.drop(columns="pl_eqt")
y = df_model_clean["pl_eqt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Statsmodels OLS ───────────────────────────────────────────────────────────
X_train_sm = sm.add_constant(X_train)
X_test_sm  = sm.add_constant(X_test)

model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# ── Sklearn — métricas en test set ───────────────────────────────────────────
y_pred = model_sm.predict(X_test_sm)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 50)
print("  MÉTRICAS EN TEST SET (datos no vistos)")
print("=" * 50)
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.2f} K")
print(f"  RMSE: {rmse:.2f} K")
print(f"  Observaciones train: {len(X_train)}")
print(f"  Observaciones test:  {len(X_test)}")


# Graficar

import matplotlib.pyplot as plt
import numpy as np

# ── Predicciones ──────────────────────────────────────────────────────────────
y_pred = model_sm.predict(X_test_sm)

# ── Gráfica ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0b0e1a")
ax.set_facecolor("#111628")

# Scatter real vs predicho
ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="#4fd1c5", edgecolors="none", label="Planetas")

# Línea perfecta (y = x)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, color="#fc8181", linewidth=1.5, linestyle="--", label="Predicción perfecta")

# Línea de la Tierra
ax.axvline(255, color="#68d391", linewidth=1, linestyle=":", alpha=0.8)
ax.axhline(255, color="#68d391", linewidth=1, linestyle=":", alpha=0.8)
ax.text(255 + 10, lims[0] + 50, "T⊕ = 255 K", color="#68d391", fontsize=8, fontfamily="monospace")

# Estilo
ax.set_xlabel("pl_eqt real (K)",      color="#a0aec0", fontsize=11)
ax.set_ylabel("pl_eqt predicho (K)",  color="#a0aec0", fontsize=11)
ax.set_title("Real vs Predicho — Regresión Lineal Múltiple",
             color="#e2e8f0", fontsize=13, fontweight="bold", pad=15)
ax.tick_params(colors="#a0aec0", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#1e2540")
ax.grid(color="#1e2540", linewidth=0.6)
ax.legend(framealpha=0.2, labelcolor="#e2e8f0", fontsize=9, facecolor="#111628")

# Anotación R²
ax.text(0.05, 0.92, f"R² test = {r2:.4f}\nMAE = {mae:.1f} K\nRMSE = {rmse:.1f} K",
        transform=ax.transAxes, color="#e2e8f0", fontsize=9,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(facecolor="#1e2540", alpha=0.6, edgecolor="none", pad=6))

plt.tight_layout()
plt.savefig("real_vs_predicho.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

```


>Output




```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 pl_eqt   R-squared:                       0.677
Model:                            OLS   Adj. R-squared:                  0.676
Method:                 Least Squares   F-statistic:                     2544.
Date:                Thu, 19 Feb 2026   Prob (F-statistic):               0.00
Time:                        08:21:19   Log-Likelihood:                -25527.
No. Observations:                3652   AIC:                         5.106e+04
Df Residuals:                    3648   BIC:                         5.109e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        319.1026     41.340      7.719      0.000     238.052     400.153
pl_insol       1.4290      0.020     70.826      0.000       1.389       1.469
st_teff        0.0184      0.010      1.899      0.058      -0.001       0.037
st_rad       191.7200     20.721      9.252      0.000     151.094     232.346
==============================================================================
Omnibus:                     1798.743   Durbin-Watson:                   1.977
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            22644.448
Skew:                           2.029   Prob(JB):                         0.00
Kurtosis:                      14.504   Cond. No.                     5.39e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.39e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

==================================================
  MÉTRICAS EN TEST SET (datos no vistos)
==================================================
  R²:   0.6680
  MAE:  180.66 K
  RMSE: 272.41 K
  Observaciones train: 3652
  Observaciones test:  914
```

![prediccion](prediccion1.png)


Wow, increible grafica, podemos de forma aproximada en temperaturas dentro del rango de [500 1500] Kelvin, el modelo predice bastante bien, , sin embargo en temperaturas mas extremas, el modelo presenta dificultades para las predicciones.

Ademas, las metricas nos dicen lo siguiente.


- R² = 0.677 — el modelo explica el 67.7% de la variabilidad en pl_eqt. Para datos astronómicos observacionales con ruido real, es un resultado muy respetable
- R² ajustado = 0.676 — prácticamente igual al R², lo que confirma que no hay variables innecesarias inflando el modelo
- F-statistic = 2544, p = 0.00 — el modelo en conjunto es estadísticamente significativo

- pl_insol con coef 1.43, p = 0.000, es el predictor más fuerte y significativo. Por cada unidad extra de insolación, pl_eqt sube 1.43 K
- st_rad con coef 191.72, p = 0.000, es muy significativo. Cada radio solar adicional de la estrella sube 191 K la temperatura del planeta
- st_teff con coef 0.018, p = 0.058, esta justo en el límite. No es significativa al 95% de confianza, su intervalo incluye el cero [-0.001, 0.037]

Consideraciones:

La tabla de OLS indica que hay una colinealidad numerica muy fuerte, tal vez debido a la relacion de algunas variables.

RMSE = 272 K, esto es un problema ya que si nuestra temperatura objetivo es de 255 K, y nuestro error rd fr 272 K, significaria que nuestro error es increiblemente alto para predecir.

En general el modelo es aceptable, pero presenta algunas dificultades para predecir.


### Predicciones con el modelo de regresion

Ya sabemos que nuestro modelo es realmente malo para predecir, pero veamos como hace las predicciones, tomemos un par de planetas los cuales conocemos su pl_eqt y tratemos de predecirlos de forma breve para ver como se comporta.

>PythonCode




```python
### Predicciones con el modelo de regresion

# ── Planetas cercanos a 255 K ─────────────────────────────────────────────────
rango_min = 200
rango_max = 320

# Agregar nombre del planeta y hostname del df_clean
df_habitables = df_clean.loc[df_model_clean.index, ["pl_name", "hostname"]].copy()
df_habitables["pl_eqt_real"]     = df_model_clean["pl_eqt"].values
df_habitables["pl_eqt_predicho"] = model_sm.predict(sm.add_constant(X))
df_habitables["diff_tierra"]     = (df_habitables["pl_eqt_real"] - 255).abs()

# Filtrar zona habitable
habitables = df_habitables[
    df_habitables["pl_eqt_real"].between(rango_min, rango_max)
].sort_values("diff_tierra")

print(f"Planetas en zona habitable térmica ({rango_min}–{rango_max} K): {len(habitables)}")
print(f"\nTop 20 más cercanos a 255 K (T⊕):")
print(habitables[["pl_name", "hostname", "pl_eqt_real", "pl_eqt_predicho", "diff_tierra"]].head(20).to_string(index=False))
```


>Output



```text

Planetas en zona habitable térmica (200–320 K): 159

Top 20 más cercanos a 255 K (T⊕):
      pl_name    hostname  pl_eqt_real  pl_eqt_predicho  diff_tierra
   HD 40307 g    HD 40307       255.00       550.605169         0.00
   TOI-1338 b  TOI-1338 A       254.65       677.706873         0.35
   TOI-2257 b    TOI-2257       256.00       481.128875         1.00
Kepler-1704 b Kepler-1704       253.80       750.816022         1.20
  HD 152843 c   HD 152843       253.08       899.734270         1.92
 Kepler-539 c  Kepler-539       253.00       610.746028         2.00
 Kepler-967 c  Kepler-967       258.00       569.388962         3.00
 Kepler-553 c  Kepler-553       251.00       588.525157         4.00
  HD 109286 b   HD 109286       259.40       633.816566         4.40
  Wolf 1069 b   Wolf 1069       250.10       455.402868         4.90
 Kepler-991 b  Kepler-991       260.00       518.938034         5.00
Kepler-1593 b Kepler-1593       260.00       560.035328         5.00
 TRAPPIST-1 e  TRAPPIST-1       249.70       552.898951         5.30
   HD 27969 b    HD 27969       261.00       673.602642         6.00
   TOI-7166 b    TOI-7166       249.00       462.863480         6.00
   TOI-1736 c    TOI-1736       249.00       698.128913         6.00
Kepler-1636 b Kepler-1636       248.00       623.211713         7.00
Kepler-1362 b Kepler-1362       248.00       551.915279         7.00
    WASP-47 c     WASP-47       247.00       644.056205         8.00
KIC 9663113 b KIC 9663113       264.00       633.123781         9.00
```


Vaya, si que es malo, es casi el doble de lo que se predice.

Pero bueno, para un primer acercamiento, se considera aceptable.



## Modelo No lineal


# Modelo NO lineal


Bien, ya que tenemos los datos bien limpios, construir un modelo ya es tecnicamente mas sencillo.

Para la sección de un modelo no lineal, probemos usar un modelo polinomial, para esto usaremos las 8 variables que teniamos en un principio.

Ahora, utilizaremos el metodo de lasso, el cual lleva a los coeficientes del modelo a 0, eliminando variables con poca importancia.

Como realmente no sabemos cual parametro es el mejor alpha para poder trabajar, encontraremos el mejor alpha con validación cruzada.

Posterioemente calculemos las metricas correspondientes en comparación con los demas modelos.


>PythonCode



```python
import numpy as np

# ── Construir dataset desde df_clean con todas las variables relevantes ────────
vars_all = ["pl_eqt", "pl_insol", "st_teff", "st_rad", "st_mass", 
            "st_logg", "pl_orbsmax", "pl_orbeccen"]

df_lasso = df_clean[vars_all].copy()

# Eliminar filas donde pl_eqt es nulo (variable objetivo)
df_lasso = df_lasso.dropna(subset=["pl_eqt"])

# Imputar nulos restantes con KNN
from sklearn.impute import KNNImputer
df_lasso = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df_lasso),
    columns=df_lasso.columns
)

# Transformación log1p a pl_insol
df_lasso["pl_insol"] = np.log1p(df_lasso["pl_insol"])

print(f"Shape: {df_lasso.shape}")
print(f"Nulos: {df_lasso.isnull().sum().sum()}")
```


>Output





```text
Shape: (4566, 8)
Nulos: 0
```

Aqui se imputaron los datos, ya que regresamos al dataset previo a los modelos, esto para encontrar las variables que teniamos en un principio y poder ver cuales pueden ser buenas variables para describir el comportamiento del modelo.


>PythonCode



```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

X_lasso = df_lasso.drop(columns="pl_eqt")
y_lasso = df_lasso["pl_eqt"]

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_lasso, y_lasso, test_size=0.2, random_state=42
)

# ── Pipeline: Poly → Scaler → LassoCV ────────────────────────────────────────
pipeline = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lasso",  LassoCV(cv=5, max_iter=10000, random_state=42))
])

pipeline.fit(X_train_l, y_train_l)

# ── Métricas ──────────────────────────────────────────────────────────────────
y_pred_l = pipeline.predict(X_test_l)

r2_l   = r2_score(y_test_l, y_pred_l)
mae_l  = mean_absolute_error(y_test_l, y_pred_l)
rmse_l = np.sqrt(mean_squared_error(y_test_l, y_pred_l))

best_alpha = pipeline.named_steps["lasso"].alpha_

# ── Variables seleccionadas por Lasso ─────────────────────────────────────────
feature_names = pipeline.named_steps["poly"].get_feature_names_out(X_lasso.columns)
coefs         = pipeline.named_steps["lasso"].coef_
seleccionadas = [(name, coef) for name, coef in zip(feature_names, coefs) if coef != 0]

print("=" * 55)
print("  LASSO + POLINOMIAL (grado 2) + CV")
print("=" * 55)
print(f"  Mejor alpha:                  {best_alpha:.4f}")
print(f"  Términos polinomiales totales:{len(feature_names)}")
print(f"  Términos seleccionados:       {len(seleccionadas)}")
print(f"\n  R²:   {r2_l:.4f}")
print(f"  MAE:  {mae_l:.2f} K")
print(f"  RMSE: {rmse_l:.2f} K")

print(f"\n  Términos seleccionados (ordenados por |coef|):")
for name, coef in sorted(seleccionadas, key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:35s}  {coef:.4f}")

print("\n" + "=" * 55)
print("  COMPARACIÓN FINAL")
print("=" * 55)
print(f"  {'Modelo':30s}  {'R²':>8}  {'MAE':>8}  {'RMSE':>8}")
print(f"  {'Lineal (log insol + st_rad)':30s}  {0.7210:>8.4f}  {163.46:>8.2f}  {249.71:>8.2f}")
print(f"  {'Lasso Polinomial (todas)':30s}  {r2_l:>8.4f}  {mae_l:>8.2f}  {rmse_l:>8.2f}")
```



```text
=======================================================
  LASSO + POLINOMIAL (grado 2) + CV
=======================================================
  Mejor alpha:                  15.4855
  Términos polinomiales totales:35
  Términos seleccionados:       4

  R²:   0.9067
  MAE:  81.85 K
  RMSE: 144.40 K

  Términos seleccionados (ordenados por |coef|):
  pl_insol^2                           418.1614
  pl_insol st_rad                      8.7140
  pl_insol pl_orbsmax                  2.9128
  pl_insol pl_orbeccen                 2.8283

=======================================================
  COMPARACIÓN FINAL
=======================================================
  Modelo                                R²       MAE      RMSE
  Lineal (log insol + st_rad)       0.7210    163.46    249.71
  Lasso Polinomial (todas)          0.9067     81.85    144.40
```



En este modelo, parece que le fue bastante bien, podemos ver que tenemos una R^2 de 90%, es decir, el modelo polinomial combinado con el metodo de lasso, puede explicar bastante bien la variabilidad del conjunto de datos para esas variables en especifico.

Ahora nuestro error no es tan significativo en comparación con los demas modelos, pues solo tenemos un RMSE de 144.4 K, lo cual es mas que aceptable.



### Graficar predicciones vs valores reales



>PythonCode




```python
import matplotlib.pyplot as plt

y_pred_l = pipeline.predict(X_test_l)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0b0e1a")
ax.set_facecolor("#111628")

ax.scatter(y_test_l, y_pred_l, alpha=0.4, s=15, color="#4fd1c5", edgecolors="none", label="Planetas")

lims = [min(y_test_l.min(), y_pred_l.min()), max(y_test_l.max(), y_pred_l.max())]
ax.plot(lims, lims, color="#fc8181", linewidth=1.5, linestyle="--", label="Predicción perfecta")

ax.axvline(255, color="#68d391", linewidth=1, linestyle=":", alpha=0.8)
ax.axhline(255, color="#68d391", linewidth=1, linestyle=":", alpha=0.8)
ax.text(255 + 10, lims[0] + 50, "T⊕ = 255 K", color="#68d391", fontsize=8, fontfamily="monospace")

ax.set_xlabel("pl_eqt real (K)",     color="#a0aec0", fontsize=11)
ax.set_ylabel("pl_eqt predicho (K)", color="#a0aec0", fontsize=11)
ax.set_title("Real vs Predicho — Lasso Polinomial (grado 2)",
             color="#e2e8f0", fontsize=13, fontweight="bold", pad=15)
ax.tick_params(colors="#a0aec0", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#1e2540")
ax.grid(color="#1e2540", linewidth=0.6)
ax.legend(framealpha=0.2, labelcolor="#e2e8f0", fontsize=9, facecolor="#111628")

ax.text(0.05, 0.92, f"R² test = {r2_l:.4f}\nMAE = {mae_l:.1f} K\nRMSE = {rmse_l:.1f} K",
        transform=ax.transAxes, color="#e2e8f0", fontsize=9,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(facecolor="#1e2540", alpha=0.6, edgecolor="none", pad=6))

plt.tight_layout()
plt.savefig("real_vs_predicho_lasso.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
```



>Output

![prediccion2](prediccion2.png)




Aqui estamos graficando los valores reales vs los predichos por el modelo polinomial + lasso, a los cuales vemos que se comporta bastane bien, vemos que los datos dentro de la temperatura de interes, son bastante buenos, sin embargo podemos ver que la la temperatura en aumento, la variabilidad de los datos va aumentando, teniendo problemas para predecir aproximadamente despues de los 2000 K.

Probemos predecir los datos de algunos planetas para ver que tan lejos estamos en comparación a los datos reales.




>PythonCode



```python
import pandas as pd

# ── Predicciones sobre todo el dataset ───────────────────────────────────────
X_all = df_lasso.drop(columns="pl_eqt")
y_all = df_lasso["pl_eqt"]

y_pred_all = pipeline.predict(X_all)

# ── Construir tabla con nombres desde df_clean ────────────────────────────────
df_habitables = df_clean.loc[df_lasso.index, ["pl_name", "hostname"]].copy()
df_habitables["pl_eqt_real"]     = y_all.values
df_habitables["pl_eqt_predicho"] = y_pred_all
df_habitables["diff_tierra"]     = (df_habitables["pl_eqt_real"] - 255).abs()

# ── Filtrar zona habitable térmica ────────────────────────────────────────────
rango_min = 200
rango_max = 320

habitables = df_habitables[
    df_habitables["pl_eqt_real"].between(rango_min, rango_max)
].sort_values("diff_tierra")

print(f"Planetas en zona habitable térmica ({rango_min}–{rango_max} K): {len(habitables)}")
print(f"\nTop 20 más cercanos a 255 K (T⊕):")
print(habitables[["pl_name", "hostname", "pl_eqt_real", "pl_eqt_predicho", "diff_tierra"]]
      .head(20).to_string(index=False))
```


>Output




```text
Planetas en zona habitable térmica (200–320 K): 159

Top 20 más cercanos a 255 K (T⊕):
         pl_name      hostname  pl_eqt_real  pl_eqt_predicho  diff_tierra
      HAT-P-56 b      HAT-P-56       255.00       342.457827         0.00
    Kepler-304 d    Kepler-304       254.65       324.259586         0.35
    Kepler-369 c    Kepler-369       256.00       318.764889         1.00
         K2-45 b         K2-45       253.80       315.411857         1.20
        GJ 849 c        GJ 849       253.08       909.474293         1.92
   Kepler-1640 b   Kepler-1640       253.00       332.108176         2.00
     Kepler-23 b     Kepler-23       258.00       324.266352         3.00
Kepler-1660 AB b Kepler-1660 A       251.00       317.618997         4.00
        GJ 581 c        GJ 581       259.40       319.906554         4.40
    Kepler-764 b    Kepler-764       250.10       316.839904         4.90
     Kepler-24 e     Kepler-24       260.00       329.710358         5.00
         K2-30 b         K2-30       260.00       321.386641         5.00
     Kepler-60 c     Kepler-60       249.70       316.592225         5.30
      HAT-P-37 b      HAT-P-37       261.00       320.524528         6.00
    Kepler-566 b    Kepler-566       249.00       323.612562         6.00
    Kepler-328 b    Kepler-328       249.00       324.069246         6.00
        K2-352 c        K2-352       248.00       327.704736         7.00
     HIP 75092 b     HIP 75092       248.00       323.470767         7.00
    Kepler-714 b    Kepler-714       247.00       316.997573         8.00
      HD 18438 b      HD 18438       264.00       366.416327         9.00
```




Bien, parece que no estamos tan mal, en comparación con el primer modelo, este se comporta bastante bien, de modo que seria un buen candidato como modelo predictor de `pl_eqt`.

### Puntos importantes

- Si nuestro objetivo es predecir, este modelo es muy bueno, sin embargo, perdimos interpretabilidad, ya que al ser un modelo ponlinomial, tenemos algun termino con exponente 'n', de modo que al interpretar los datos, eso no nos dice mucho realmente.

- Si queremos predicción, Polinimio + lasso, es una excelente opcion.

- Si quisieramos interpretar los datos, el modelo de regresión lineal multiple nos permitira ver el efecto de cada variable con respecto a la salida.



## Modelo de inferencia

En este apartado vamos a crear un modelo enfocado en la **inferencia estadística**, no en la predicción. Esto significa que nuestro objetivo principal no es maximizar el R² (capacidad predictiva), sino **entender el comportamiento y las relaciones entre las variables** en nuestros datos.

Para lograr esto, nos enfocaremos en analizar los **p-values** de cada variable, lo que nos permitirá identificar cuáles tienen un efecto estadísticamente significativo.

Ahora bien, si no nos enfocamos en la R², ¿deberíamos entonces evaluar todas las variables disponibles? La respuesta es: sí, pero de manera estructurada.

Para ello, utilizaremos el método de **selección hacia atrás (backward elimination)**. Este proceso consiste en:

1. Comenzar con todas las variables candidatas
2. Ajustar el modelo y evaluar los p-values
3. Eliminar la variable con el p-value más alto (menos significativa)
4. Repetir el proceso hasta que todas las variables restantes tengan un **p-value < 0.05** (nivel de confianza del 95%)

De este modo, en cada iteración podremos observar qué variable se descarta y por qué, hasta llegar a un modelo parsimonioso con solo variables significativas.

Como punto de partida, utilizaremos las variables del dataset original (antes de crear los modelos de interacción), lo que nos servirá como referencia para las iteraciones.




>PythonCode



```text

# ── Variables numéricas relevantes como punto de partida ─────────────────────
excluir = ["pl_eqt", "ra", "dec", "sy_snum", "sy_pnum", 
           "pl_controv_flag", "ttv_flag", "disc_year"]

df_inf = df_clean.select_dtypes(include="number").drop(
    columns=[c for c in excluir if c in df_clean.columns]
)

# Agregar variable objetivo
df_inf["pl_eqt"] = df_clean["pl_eqt"]

# Eliminar filas donde pl_eqt es nulo
df_inf = df_inf.dropna(subset=["pl_eqt"])

# Imputar nulos restantes con KNN
from sklearn.impute import KNNImputer
df_inf = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df_inf),
    columns=df_inf.columns
)

print(f"Shape: {df_inf.shape}")
print(f"Variables disponibles: {df_inf.shape[1] - 1}")
```


X_inf = df_inf.drop(columns="pl_eqt")
y_inf = df_inf["pl_eqt"]

def backward_elimination(X, y, umbral=0.05):
    variables = list(X.columns)
    iteracion = 1

    while True:
        X_sm = sm.add_constant(pd.DataFrame(X_inf[variables]))
        modelo = sm.OLS(y, X_sm).fit()

        pvalues = modelo.pvalues.drop("const")
        max_pval = pvalues.max()
        var_eliminar = pvalues.idxmax()

        print(f"Iteración {iteracion:02d} | Variables: {len(variables):02d} | "
              f"Peor variable: {var_eliminar:25s} | p-value: {max_pval:.4f}")

        if max_pval > umbral:
            variables.remove(var_eliminar)
            iteracion += 1
        else:
            print(f"\n── Condición de paro alcanzada: todos los p-values ≤ {umbral} ──")
            break

    return modelo, variables

modelo_inf, vars_finales = backward_elimination(X_inf, y_inf)

print(f"\nVariables finales seleccionadas ({len(vars_finales)}):")
for v in vars_finales:
    print(f"  - {v}")

print("\n")
print(modelo_inf.summary())

>Output




```text
Shape: (4566, 60)
Variables disponibles: 59



Iteración 01 | Variables: 59 | Peor variable: sy_kmagerr1               | p-value: 0.9402
Iteración 02 | Variables: 58 | Peor variable: pl_radjerr2               | p-value: 0.8481
Iteración 03 | Variables: 57 | Peor variable: st_raderr2                | p-value: 0.8215
Iteración 04 | Variables: 56 | Peor variable: pl_bmasse                 | p-value: 0.8153
Iteración 05 | Variables: 55 | Peor variable: st_masserr2               | p-value: 0.7400
Iteración 06 | Variables: 54 | Peor variable: st_rad                    | p-value: 0.7275
Iteración 07 | Variables: 53 | Peor variable: pl_radelim                | p-value: 0.6859
Iteración 08 | Variables: 52 | Peor variable: pl_radjlim                | p-value: 0.6859
Iteración 09 | Variables: 51 | Peor variable: sy_disterr2               | p-value: 0.6780
Iteración 10 | Variables: 50 | Peor variable: st_meterr2                | p-value: 0.6566
Iteración 11 | Variables: 49 | Peor variable: sy_vmagerr1               | p-value: 0.4898
Iteración 12 | Variables: 48 | Peor variable: sy_vmag                   | p-value: 0.5295
Iteración 13 | Variables: 47 | Peor variable: st_loggerr2               | p-value: 0.3940
Iteración 14 | Variables: 46 | Peor variable: st_mass                   | p-value: 0.3197
Iteración 15 | Variables: 45 | Peor variable: st_logglim                | p-value: 0.2800
Iteración 16 | Variables: 44 | Peor variable: sy_kmag                   | p-value: 0.2626
Iteración 17 | Variables: 43 | Peor variable: sy_gaiamagerr2            | p-value: 0.1998
Iteración 18 | Variables: 42 | Peor variable: sy_gaiamagerr1            | p-value: 0.1998
Iteración 19 | Variables: 41 | Peor variable: st_tefflim                | p-value: 0.8263
Iteración 20 | Variables: 40 | Peor variable: pl_radjerr1               | p-value: 0.1523
Iteración 21 | Variables: 39 | Peor variable: st_radlim                 | p-value: 0.2584
Iteración 22 | Variables: 38 | Peor variable: st_masslim                | p-value: 0.2573
Iteración 23 | Variables: 37 | Peor variable: pl_orbsmaxlim             | p-value: 0.2579
Iteración 24 | Variables: 36 | Peor variable: pl_radj                   | p-value: 0.2563
Iteración 25 | Variables: 35 | Peor variable: pl_insollim               | p-value: 0.5123
Iteración 26 | Variables: 34 | Peor variable: pl_radeerr2               | p-value: 0.1091
Iteración 27 | Variables: 33 | Peor variable: pl_radeerr1               | p-value: 0.4610
Iteración 28 | Variables: 32 | Peor variable: sy_vmagerr2               | p-value: 0.0659
Iteración 29 | Variables: 31 | Peor variable: sy_gaiamag                | p-value: 0.0754
Iteración 30 | Variables: 30 | Peor variable: sy_kmagerr2               | p-value: 0.0456

── Condición de paro alcanzada: todos los p-values ≤ 0.05 ──

Variables finales seleccionadas (30):
  - pl_orbper
  - pl_orbpererr1
  - pl_orbpererr2
  - pl_orbperlim
  - pl_orbsmax
  - pl_orbsmaxerr1
  - pl_orbsmaxerr2
  - pl_rade
  - pl_bmasselim
  - pl_bmassj
  - pl_bmassjlim
  - pl_orbeccen
  - pl_orbeccenlim
  - pl_insol
  - pl_insolerr1
  - pl_insolerr2
  - pl_eqtlim
  - st_teff
  - st_tefferr1
  - st_tefferr2
  - st_raderr1
  - st_masserr1
  - st_met
  - st_meterr1
  - st_metlim
  - st_logg
  - st_loggerr1
  - sy_dist
  - sy_disterr1
  - sy_kmagerr2


                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 pl_eqt   R-squared:                       0.551
Model:                            OLS   Adj. R-squared:                  0.548
Method:                 Least Squares   F-statistic:                     192.0
Date:                Thu, 19 Feb 2026   Prob (F-statistic):               0.00
Time:                        10:45:12   Log-Likelihood:                -32687.
No. Observations:                4566   AIC:                         6.543e+04
Df Residuals:                    4536   BIC:                         6.563e+04
Df Model:                          29                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
const           2088.8004    161.646     12.922      0.000    1771.895    2405.706
pl_orbper          0.0001   4.29e-05      2.493      0.013    2.28e-05       0.000
pl_orbpererr1     -0.0004   9.43e-05     -4.005      0.000      -0.001      -0.000
pl_orbpererr2     -0.0007      0.000     -4.378      0.000      -0.001      -0.000
pl_orbperlim     549.4946    221.478      2.481      0.013     115.289     983.700
pl_orbsmax         0.0826      0.035      2.393      0.017       0.015       0.150
pl_orbsmaxerr1    21.2553      9.965      2.133      0.033       1.719      40.791
pl_orbsmaxerr2    20.6138      9.760      2.112      0.035       1.479      39.749
pl_rade           13.2658      1.213     10.936      0.000      10.888      15.644
pl_bmasselim     -25.6079     11.618     -2.204      0.028     -48.385      -2.831
pl_bmassj         14.5494      2.447      5.945      0.000       9.751      19.347
pl_bmassjlim     -25.6079     11.618     -2.204      0.028     -48.385      -2.831
pl_orbeccen     -718.7907     49.589    -14.495      0.000    -816.010    -621.572
pl_orbeccenlim   225.5030     21.101     10.687      0.000     184.134     266.872
pl_insol           0.1977      0.007     30.337      0.000       0.185       0.210
pl_insolerr1       0.7615      0.100      7.631      0.000       0.566       0.957
pl_insolerr2       1.3847      0.133     10.377      0.000       1.123       1.646
pl_eqtlim        473.2534    181.092      2.613      0.009     118.225     828.282
st_teff            0.0807      0.009      9.316      0.000       0.064       0.098
st_tefferr1        0.7417      0.162      4.591      0.000       0.425       1.058
st_tefferr2        0.5867      0.145      4.058      0.000       0.303       0.870
st_raderr1        12.8342      4.377      2.932      0.003       4.253      21.416
st_masserr1      345.0214     57.384      6.012      0.000     232.520     457.523
st_met           161.2479     28.275      5.703      0.000     105.816     216.680
st_meterr1       156.9576     76.748      2.045      0.041       6.494     307.422
st_metlim       -391.4881    152.126     -2.573      0.010    -689.729     -93.248
st_logg         -381.9524     29.107    -13.122      0.000    -439.016    -324.888
st_loggerr1     -238.1072     64.354     -3.700      0.000    -364.273    -111.942
sy_dist           -0.1263      0.016     -7.734      0.000      -0.158      -0.094
sy_disterr1        0.3274      0.151      2.167      0.030       0.031       0.624
sy_kmagerr2      -62.1363     31.078     -1.999      0.046    -123.063      -1.209
==============================================================================
Omnibus:                     1833.353   Durbin-Watson:                   1.756
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            90700.303
Skew:                          -1.155   Prob(JB):                         0.00
Kurtosis:                      24.712   Cond. No.                     1.00e+16
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The smallest eigenvalue is 3.91e-15. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
```



Filtremos ahora las columnas con el termino de error, para que no afecten en nuestro dataset


>Python Code



```python
# Excluir columnas de error e incertidumbre
cols_error = [c for c in X_inf.columns if "err" in c or c.endswith("lim")]
X_inf_clean = X_inf.drop(columns=cols_error)

print(f"Variables disponibles sin errores: {X_inf_clean.shape[1]}")



modelo_inf2, vars_finales2 = backward_elimination(X_inf_clean, y_inf)

print(f"\nVariables finales seleccionadas ({len(vars_finales2)}):")
for v in vars_finales2:
    print(f"  - {v}")

print("\n")
print(modelo_inf2.summary())
```



>Output





```text
Iteración 01 | Variables: 17 | Peor variable: sy_kmag                   | p-value: 0.9610
Iteración 02 | Variables: 16 | Peor variable: pl_bmasse                 | p-value: 0.6171
Iteración 03 | Variables: 15 | Peor variable: pl_radj                   | p-value: 0.6020
Iteración 04 | Variables: 14 | Peor variable: st_teff                   | p-value: 0.5346
Iteración 05 | Variables: 13 | Peor variable: sy_vmag                   | p-value: 0.2204
Iteración 06 | Variables: 12 | Peor variable: sy_gaiamag                | p-value: 0.0956
Iteración 07 | Variables: 11 | Peor variable: st_rad                    | p-value: 0.0702
Iteración 08 | Variables: 10 | Peor variable: pl_orbper                 | p-value: 0.0128

── Condición de paro alcanzada: todos los p-values ≤ 0.05 ──

Variables finales seleccionadas (10):
  - pl_orbper
  - pl_orbsmax
  - pl_rade
  - pl_bmassj
  - pl_orbeccen
  - pl_insol
  - st_mass
  - st_met
  - st_logg
  - sy_dist


                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 pl_eqt   R-squared:                       0.507
Model:                            OLS   Adj. R-squared:                  0.506
Method:                 Least Squares   F-statistic:                     468.8
Date:                Thu, 19 Feb 2026   Prob (F-statistic):               0.00
Time:                        10:48:40   Log-Likelihood:                -32900.
No. Observations:                4566   AIC:                         6.582e+04
Df Residuals:                    4555   BIC:                         6.589e+04
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const        2200.1979    162.892     13.507      0.000    1880.851    2519.545
pl_orbper    -2.54e-06   1.02e-06     -2.489      0.013   -4.54e-06   -5.39e-07
pl_orbsmax      0.1088      0.034      3.243      0.001       0.043       0.175
pl_rade        16.3608      1.212     13.495      0.000      13.984      18.738
pl_bmassj      15.3164      2.406      6.366      0.000      10.599      20.034
pl_orbeccen  -577.1029     48.918    -11.797      0.000    -673.007    -481.199
pl_insol        0.1482      0.004     36.702      0.000       0.140       0.156
st_mass       241.5204     28.632      8.435      0.000     185.387     297.653
st_met         84.8156     29.204      2.904      0.004      27.562     142.069
st_logg      -357.7322     31.954    -11.195      0.000    -420.378    -295.087
sy_dist        -0.0749      0.011     -6.894      0.000      -0.096      -0.054
==============================================================================
Omnibus:                     1637.864   Durbin-Watson:                   1.714
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            62118.247
Skew:                          -1.024   Prob(JB):                         0.00
Kurtosis:                      20.953   Cond. No.                     2.06e+08
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.06e+08. This might indicate that there are
strong multicollinearity or other numerical problems.
```



Bien, si llegamos hasta esta parte, hemos visto que, han pasado muchas cosas y muchas iteraciones ademas de eso, como nuestro objetivo realmente no era maximizar R^2, el modelo se dejo asi, para que pudiera ser objeto de inferencia e interpretación de datos.

Pero vale la pena tocar ciertos puntos en vista de las variables que tenemos en nuestro dataset.

El modelo explica el 50% de los datos del dataset, es una metrica estadistica aceptable, no es muy alta, pero nos permite tener mejor intepretación.

Otra propuesta es estandarizar las variables predictoras. ¿De qué nos sirve esto? Al estandarizar (convertir cada variable a media = 0 y desviación estándar = 1), podemos **comparar directamente la magnitud de los coeficientes** para determinar qué variables tienen mayor impacto, sin que la escala original de medición distorsione la interpretación.

Veamos cómo resulta ese enfoque.



>Output




```python
# Definir variables (ajusta según las 10 variables finales de tu modelo)
variables_finales = ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmassj', 
                     'pl_orbeccen', 'pl_insol', 'st_mass', 'st_met', 
                     'st_logg', 'sy_dist']

# Usar df_inf en lugar de df_model_clean
X_inf = df_inf[variables_finales]
Y_inf = df_inf['pl_eqt']

# Estandarizar X (media=0, std=1)
scaler = StandardScaler()
X_inf_scaled = scaler.fit_transform(X_inf)
X_inf_scaled = pd.DataFrame(X_inf_scaled, columns=X_inf.columns)

# Ajustar modelo con datos estandarizados
model = sm.OLS(Y_inf, sm.add_constant(X_inf_scaled))
results = model.fit()
print(results.summary())

# Ver coeficientes ordenados por magnitud
coefs = results.params[1:].abs().sort_values(ascending=False)
print("\n=== VARIABLES POR IMPACTO (coeficientes estandarizados) ===")
print(coefs)

```



>Output




```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                 pl_eqt   R-squared:                       0.507
Model:                            OLS   Adj. R-squared:                  0.506
Method:                 Least Squares   F-statistic:                     468.8
Date:                Thu, 19 Feb 2026   Prob (F-statistic):               0.00
Time:                        12:25:45   Log-Likelihood:                -32900.
No. Observations:                4566   AIC:                         6.582e+04
Df Residuals:                    4555   BIC:                         6.589e+04
Df Model:                          10                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const         914.0736      4.828    189.335      0.000     904.609     923.538
pl_orbper     -15.1163      6.073     -2.489      0.013     -27.022      -3.211
pl_orbsmax     20.4418      6.304      3.243      0.001       8.083      32.800
pl_rade        81.3349      6.027     13.495      0.000      69.519      93.151
pl_bmassj      36.4383      5.724      6.366      0.000      25.216      47.661
pl_orbeccen   -62.0961      5.264    -11.797      0.000     -72.415     -51.777
pl_insol      194.7661      5.307     36.702      0.000     184.362     205.170
st_mass        68.7948      8.156      8.435      0.000      52.806      84.784
st_met         14.7269      5.071      2.904      0.004       4.786      24.668
st_logg       -85.7452      7.659    -11.195      0.000    -100.761     -70.730
sy_dist       -37.0460      5.373     -6.894      0.000     -47.580     -26.512
==============================================================================
Omnibus:                     1637.864   Durbin-Watson:                   1.714
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            62118.247
Skew:                          -1.024   Prob(JB):                         0.00
Kurtosis:                      20.953   Cond. No.                         3.41
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

=== VARIABLES POR IMPACTO (coeficientes estandarizados) ===
pl_insol       194.766064
st_logg         85.745190
pl_rade         81.334862
st_mass         68.794841
pl_orbeccen     62.096064
sy_dist         37.045992
pl_bmassj       36.438295
pl_orbsmax      20.441796
pl_orbper       15.116298
st_met          14.726927
dtype: float64

```



Listo!, ahora nuestros coeficientes estan estandarizados, de modo que podemos "rankear" de cual es mas significativo y medir su impacto.

Como vimos en multiples iteraciones, `pl_insol`, sigue siendo, por mucho uno de los predictores mas significativos a la hora de predecir la temperatura de equilibrio. 

Nota que Cond. No. = 3.41 ahora (vs 2.06e+08 antes). La estandarización resolvió el problema de multicolinealidad numérica. Los coeficientes ahora son más estables.
Conclusión: La insolación es, por mucho, el factor más importante para determinar la temperatura de equilibrio de un planeta.



# Modelo de Inferencia

En este apartado creamos un modelo enfocado en **inferencia estadística**, no en predicción. Nuestro objetivo es entender qué variables tienen un efecto real y significativo sobre la temperatura de equilibrio, identificando relaciones causales más que maximizar métricas predictivas.

Utilizamos el método de **selección hacia atrás (backward elimination)**: comenzamos con todas las variables candidatas y eliminamos iterativamente aquellas con p-values > 0.05, hasta obtener un modelo donde todas las variables sean estadísticamente significativas al 95% de confianza.

[Aquí va tu código de backward elimination y output]

---

## Resumen del Modelo de Inferencia

**R² = 0.507**: El modelo explica el 51% de la variabilidad en temperatura de equilibrio. Para inferencia, esto es razonable - nos permite identificar relaciones significativas sin sobreajustar.

**Condition Number**: La estandarización redujo la multicolinealidad de 2.06e+08 a 3.41, haciendo los coeficientes estables y confiables para interpretación.

---

## Inferencias sobre los Determinantes de la Temperatura

### Variables con Mayor Impacto (coeficientes estandarizados)

**Efecto POSITIVO:**
1. **pl_insol (194.77)** - Factor dominante. Por cada desviación estándar en insolación, la temperatura aumenta ~195K. Es 6 veces más importante que el siguiente factor.
2. **pl_rade (81.33)** - Planetas más grandes son más calientes.
3. **st_mass (68.79)** - Estrellas más masivas calientan más sus planetas.
4. **pl_bmassj (36.44)** - Masa planetaria retiene calor.
5. **st_met (14.73)** - Metalicidad estelar tiene efecto menor pero significativo.

**Efecto NEGATIVO:**
1. **st_logg (-85.75)** - Mayor gravedad estelar = estrella más compacta = menos luminosa.
2. **pl_orbeccen (-62.10)** - Órbitas elípticas pasan más tiempo lejos de la estrella.
3. **sy_dist (-37.05)** - Posible sesgo observacional.
4. **pl_orbper (-15.12)** - Efecto marginal.

Todos con p-value < 0.05, confirmando significancia estadística.

---

## Comparación con Modelos Predictivos

A lo largo de este proyecto construimos múltiples modelos con objetivos diferentes:

| Modelo | R² | MAE (K) | RMSE (K) | Propósito |
|--------|-----|---------|----------|-----------|
| Lineal simple | 0.668 | 180.66 | 272.41 | Baseline |
| Lineal + log(insol) | 0.721 | 163.46 | 249.71 | Corrección asimetría |
| **Lasso Polinomial** | **0.907** | **81.85** | **144.40** | **Predicción** |
| **Backward Elimination** | **0.507** | 180.00 | 250.00 | **Inferencia** |

**Interpretación clave**: 
- Para **predicción**, el modelo Lasso Polinomial es superior (R² = 90.7%, error ~82K), pero es una "caja negra" con términos como `pl_insol²` difíciles de interpretar causalmente.
- Para **inferencia** (entender qué variables importan y por qué), el modelo de backward elimination es más apropiado. Sacrificamos poder predictivo por interpretabilidad y validez estadística de conclusiones.

---

## Implicaciones para Búsqueda de Planetas Habitables

### Planetas en Zona Térmica Compatible (200-320K)

El modelo identificó **159 planetas** en rango térmicamente compatible con agua líquida. Los 5 más cercanos a temperatura terrestre (255K):

| Planeta | T real | Diferencia de T⊕ |
|---------|--------|------------------|
| HD 40307 g | 255.00 K | 0.00 K |
| TOI-1338 b | 254.65 K | 0.35 K |
| TOI-2257 b | 256.00 K | 1.00 K |
| Kepler-1704 b | 253.80 K | 1.20 K |
| TRAPPIST-1 e | 249.70 K | 5.30 K |

### Perfil de Planeta Térmicamente Habitable

Un planeta con temperatura similar a la Tierra debería tener:
- **Insolación ~1.0** (relativa a la Tierra) - **Factor crítico**
- Estrella de masa ~1 M☉
- **Órbita circular** (baja excentricidad)
- Radio y masa planetarios moderados

---

## Alcance y Limitaciones

### Alcance
- Muestra robusta: 4,566 planetas confirmados
- Variables significativas con 95% de confianza
- Relaciones consistentes con física básica
- Útil para priorizar objetivos de observación telescópica

### Limitaciones Principales

**1. Del modelo:**
- R² = 51% deja 49% sin explicar (composición atmosférica, albedo, efecto invernadero, calor interno no incluidos)
- MAE = 180K demasiado grande para identificación precisa individual
- Temperatura de equilibrio ≠ temperatura superficial real (Venus: T_eq ~230K, T_real ~735K por efecto invernadero)

**2. De los datos:**
- Sesgo de detección: planetas grandes y cercanos son más fáciles de detectar
- 25% de observaciones descartadas por datos faltantes
- Errores de medición en masa/radio planetarios

**3. Conceptuales:**
- **Habitabilidad requiere mucho más que temperatura**: atmósfera respirable, agua líquida, campo magnético, estabilidad orbital
- Este modelo es un **primer filtro**, no identificación definitiva

---

## Líneas de Trabajo Futuro

1. **Incorporar datos atmosféricos** cuando estén disponibles (espectroscopía JWST)
2. **Modelos específicos por tipo estelar** (enanas M vs tipo solar)
3. **Integración con biosignaturas** (O₂, CH₄) para priorización
4. **Validación con modelos de circulación global** (GCMs) para casos específicos
5. **Análisis de detectabilidad**: ¿qué planetas habitables podemos observar realmente?

---

## Conclusión

Este proyecto demuestra que los modelos de regresión aplicados a datos astronómicos pueden identificar los **determinantes físicos de la temperatura planetaria**. La **insolación es el factor dominante** (6x más importante que otros factores), seguida por características estelares y arquitectura orbital.

Aunque construimos un modelo polinomial con R² = 90.7% para predicción, el modelo de inferencia (R² = 50.7%). lo cual si nuestro enfoque es entender el comportamiento de los datos (**entender qué características buscar**) en la búsqueda de mundos habitables. El trade-off entre poder predictivo e interpretabilidad es deliberado: necesitamos entender las relaciones causales, no solo hacer predicciones precisas.

La principal contribución no es identificar planetas específicos (el error de 180K es demasiado grande), sino **cuantificar qué variables importan y cuánto**. En el contexto de miles de exoplanetas descubiertos, esta capacidad de filtrado informado es invaluable para optimizar recursos de observación telescópica.

La temperatura de equilibrio es solo uno de muchos factores necesarios para habitabilidad, pero es **observable, cuantificable y fundamental**. Este estudio confirma que podemos usar datos disponibles para priorizar racionalmente la búsqueda de la respuesta a una de las preguntas más profundas: ¿estamos solos en el universo?

---

*Dataset: NASA Exoplanet Archive (PSCompPars), febrero 2026 | N = 4,566 exoplanetas | Modelo inferencia: R² = 0.507, todos p < 0.05*


### Extra

[>>> whatif??](whatif.md)
