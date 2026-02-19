# Cargar base de datos

En este apartado abarcaremos un poco de contexto y la naturaleza de los datos, segun el diccionario proporcionado en la base de datos de la NASA.


> PythonCode


```python
# cargar datos

from google.colab import drive
drive.mount('/content/drive')
# ruta
df_raw = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/Proyecto 1/PSCompPars_2026.02.16_14.28.02.csv', comment='#')
df_raw
```


>Output

| pl_name | hostname | sy_snum | sy_pnum | discoverymethod | disc_year | disc_facility | pl_controv_flag | pl_orbper | pl_orbpererr1 | ... | sy_vmag | sy_kmag | sy_gaiamag |
|---------|----------|---------|---------|-----------------|-----------|---------------|-----------------|-----------|---------------|-----|---------|---------|------------|
| 11 Com b | 11 Com | 2 | 1 | Radial Velocity | 2007.0 | Xinglong Station | 0 | 323.21 | 0.06 | ... | 4.72 | 2.28 | 4.44 |
| 11 UMi b | 11 UMi | 1 | 1 | Radial Velocity | 2009.0 | Thueringer Landessternwarte | 0 | 516.22 | 3.20 | ... | 5.01 | 1.94 | 4.56 |
| 14 And b | 14 And | 1 | 1 | Radial Velocity | 2008.0 | Okayama Astrophysical | 0 | 186.76 | 0.11 | ... | 5.23 | 2.33 | 4.92 |
| 14 Her b | 14 Her | 1 | 2 | Radial Velocity | 2002.0 | W. M. Keck Observatory | 0 | 1765.04 | 1.68 | ... | 6.62 | 4.71 | 6.38 |
| 16 Cyg B b | 16 Cyg B | 3 | 1 | Radial Velocity | 1996.0 | Multiple Observatories | 0 | 798.50 | 1.00 | ... | 6.22 | 4.65 | 6.06 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| ups And b | ups And | 2 | 3 | Radial Velocity | 1996.0 | Lick Observatory | 0 | 4.62 | 0.00 | ... | 4.10 | 2.86 | 3.99 |
| ups And c | ups And | 2 | 3 | Radial Velocity | 1999.0 | Multiple Observatories | 0 | 241.26 | 0.06 | ... | 4.10 | 2.86 | 3.99 |
| ups And d | ups And | 2 | 3 | Radial Velocity | 1999.0 | Multiple Observatories | 0 | 1276.46 | 0.57 | ... | 4.10 | 2.86 | 3.99 |
| ups Leo b | ups Leo | 1 | 1 | Radial Velocity | 2021.0 | Okayama Astrophysical | 0 | 385.20 | 2.80 | ... | 4.30 | 2.18 | 4.03 |
| xi Aql b | xi Aql | 1 | 1 | Radial Velocity | 2007.0 | Okayama Astrophysical | 0 | 136.97 | 0.11 | ... | 4.71 | 2.17 | 4.43 |

**6,107 rows × 84 columns**


# Diccionario de Variables — NASA Exoplanet Archive (PSCompPars)

## Identificación del Sistema

| Variable | Descripción |
|---|---|
| `pl_name` | Nombre del planeta |
| `hostname` | Nombre de la estrella anfitriona |
| `sy_snum` | Número de estrellas en el sistema |
| `sy_pnum` | Número de planetas en el sistema |

## Descubrimiento

| Variable | Descripción |
|---|---|
| `discoverymethod` | Método de descubrimiento (ej. Radial Velocity, Transit) |
| `disc_year` | Año de descubrimiento |
| `disc_facility` | Instalación/observatorio donde fue descubierto |
| `pl_controv_flag` | Bandera de planeta controvertido (1 = controvertido) |

## Parámetros Orbitales del Planeta

| Variable | Descripción |
|---|---|
| `pl_orbper` | Período orbital (días) — *La Tierra tarda 365.25 días en orbitar el Sol* |
| `pl_orbpererr1/2` | Incertidumbre superior/inferior del período orbital |
| `pl_orbperlim` | Límite del período orbital |
| `pl_orbsmax` | Semieje mayor de la órbita en Unidades Astronómicas (UA) — *1 UA = distancia Tierra-Sol ≈ 150 millones de km* |
| `pl_orbsmaxerr1/2` | Incertidumbre del semieje mayor |
| `pl_orbsmaxlim` | Límite del semieje mayor |
| `pl_orbeccen` | Excentricidad orbital — *0 = órbita perfectamente circular; 1 = órbita parabólica (escape). La Tierra tiene ~0.017* |
| `pl_orbeccenerr1/2` | Incertidumbre de la excentricidad |
| `pl_orbeccenlim` | Límite de excentricidad |

## Radio del Planeta

| Variable | Descripción |
|---|---|
| `pl_rade` | Radio del planeta en radios terrestres (R⊕) — *1 R⊕ = 6,371 km (radio de la Tierra)* |
| `pl_radeerr1/2` | Incertidumbre del radio en R⊕ |
| `pl_radelim` | Límite del radio en R⊕ |
| `pl_radj` | Radio del planeta en radios de Júpiter (RJ) — *1 RJ = 71,492 km ≈ 11.2 R⊕* |
| `pl_radjerr1/2` | Incertidumbre del radio en RJ |
| `pl_radjlim` | Límite del radio en RJ |

## Masa del Planeta

| Variable | Descripción |
|---|---|
| `pl_bmasse` | Masa del planeta en masas terrestres (M⊕) — *1 M⊕ = 5.97 × 10²⁴ kg* |
| `pl_bmasseerr1/2` | Incertidumbre de la masa en M⊕ |
| `pl_bmasselim` | Límite de la masa en M⊕ |
| `pl_bmassj` | Masa del planeta en masas de Júpiter (MJ) — *1 MJ = 1,898 × 10²⁷ kg ≈ 318 M⊕* |
| `pl_bmassjerr1/2` | Incertidumbre de la masa en MJ |
| `pl_bmassjlim` | Límite de la masa en MJ |
| `pl_bmassprov` | Fuente de la masa (ej. `Msini` = masa mínima proyectada, `Mass` = masa real estimada) |

## Otras Propiedades del Planeta

| Variable | Descripción |
|---|---|
| `pl_insol` | Insolación recibida por el planeta relativa a la Tierra — *1 = misma radiación que recibe la Tierra. Venus recibe ~1.9, Marte ~0.43* |
| `pl_insolerr1/2` | Incertidumbre de la insolación |
| `pl_insollim` | Límite de insolación |
| `pl_eqt` | Temperatura de equilibrio del planeta (Kelvin) — *0 K = −273 °C. La Tierra tiene ~255 K (−18 °C) sin efecto invernadero. El agua líquida existe entre ~273 K y 373 K* |
| `pl_eqterr1/2` | Incertidumbre de la temperatura de equilibrio |
| `pl_eqtlim` | Límite de la temperatura de equilibrio |
| `ttv_flag` | Bandera de variaciones en el tiempo de tránsito (1 = TTV detectado; indica posible influencia gravitacional de otro planeta) |

## Propiedades de la Estrella

| Variable | Descripción |
|---|---|
| `st_spectype` | Tipo espectral de la estrella — *clasificación OBAFGKM de más caliente/masiva a más fría/pequeña. El Sol es G2* |
| `st_teff` | Temperatura efectiva de la estrella (Kelvin) — *El Sol tiene ~5,778 K. Estrellas azules superan 30,000 K; enanas rojas rondan 3,000 K* |
| `st_tefferr1/2` | Incertidumbre de la temperatura efectiva |
| `st_tefflim` | Límite de la temperatura efectiva |
| `st_rad` | Radio estelar en radios solares (R⊙) — *1 R⊙ = 696,000 km ≈ 109 radios terrestres* |
| `st_raderr1/2` | Incertidumbre del radio estelar |
| `st_radlim` | Límite del radio estelar |
| `st_mass` | Masa estelar en masas solares (M⊙) — *1 M⊙ = 1.989 × 10³⁰ kg ≈ 333,000 masas terrestres* |
| `st_masserr1/2` | Incertidumbre de la masa estelar |
| `st_masslim` | Límite de la masa estelar |
| `st_met` | Metalicidad de la estrella en dex — *mide la abundancia de elementos más pesados que el helio respecto al Sol. 0 = igual que el Sol; +0.3 = doble de metales; −0.3 = la mitad* |
| `st_meterr1/2` | Incertidumbre de la metalicidad |
| `st_metlim` | Límite de la metalicidad |
| `st_metratio` | Ratio de metalicidad utilizado (ej. `[Fe/H]` = hierro respecto al hidrógeno, el más común) |
| `st_logg` | Gravedad superficial estelar en escala logarítmica (log g, cgs) — *el Sol tiene log g ≈ 4.44. Valores altos (~4.5) indican enanas compactas; valores bajos (~2) indican gigantes* |
| `st_loggerr1/2` | Incertidumbre del log g |
| `st_logglim` | Límite del log g |

## Posición y Distancia

| Variable | Descripción |
|---|---|
| `rastr` | Ascensión recta en formato sexagesimal (horas:minutos:segundos) — *equivalente a la longitud geográfica pero en el cielo* |
| `ra` | Ascensión recta en grados decimales (0° a 360°) |
| `decstr` | Declinación en formato sexagesimal — *equivalente a la latitud geográfica en el cielo (−90° a +90°)* |
| `dec` | Declinación en grados decimales |
| `sy_dist` | Distancia al sistema en parsecs (pc) — *1 pc ≈ 3.26 años luz ≈ 30.9 billones de km. La estrella más cercana, Próxima Centauri, está a ~1.3 pc* |
| `sy_disterr1/2` | Incertidumbre de la distancia |

## Magnitudes del Sistema

| Variable | Descripción |
|---|---|
| `sy_vmag` | Magnitud aparente en banda V (luz visible) — *escala invertida: menor valor = más brillante. El Sol tiene −26.7; ojo humano ve hasta ~+6; Júpiter ronda −2* |
| `sy_vmagerr1/2` | Incertidumbre en magnitud V |
| `sy_kmag` | Magnitud en banda K (infrarrojo cercano, ~2.2 µm) — *útil para estrellas frías y objetos con polvo interestelar* |
| `sy_kmagerr1/2` | Incertidumbre en magnitud K |
| `sy_gaiamag` | Magnitud en banda G del satélite Gaia (óptico amplio, ~330–1050 nm) — *muy precisa, usada para astrometría moderna* |
| `sy_gaiamagerr1/2` | Incertidumbre en magnitud Gaia |

---
> **Nota:** Las columnas con sufijo `err1` corresponden a la incertidumbre superior (+) y `err2` a la inferior (−). Las columnas con sufijo `lim` indican si el valor es un límite superior/inferior (1) o una medición (0).
>
> **Símbolo solar ⊙:** El círculo con una cruz representa al Sol, nuestra estrella. Se usa como unidad de referencia en variables estelares: masas solares (M⊙), radios solares (R⊙), etc.
>
> **Símbolo terrestre ⊕:** El círculo con una cruz representa al Sol, nuestra estrella. Se usa como unidad de referencia en variables estelares: masas solares (M⊕), radios solares (R⊙), etc.


---

### Siguiente pagina


[>>>> Limpieza de datos](limpieza.md)
