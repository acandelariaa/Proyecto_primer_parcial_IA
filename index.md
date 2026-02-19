# 🪐 Predicción de Temperatura de Equilibrio de Exoplanetas
**Análisis de habitabilidad térmica usando modelos de regresión**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Descripción

Este proyecto utiliza modelos de regresión para predecir la **temperatura de equilibrio** de exoplanetas y determinar cuáles podrían ser potencialmente habitables según criterios térmicos (200-320 K, compatible con agua líquida).

**Dataset**: NASA Exoplanet Archive (PSCompPars) - 6,107 exoplanetas confirmados  
**Objetivo**: Identificar los determinantes físicos de la temperatura planetaria y priorizar candidatos para búsqueda de vida extraterrestre.

---

## 🎯 Resultados Principales

| Modelo | R² | MAE (K) | RMSE (K) | Uso |
|--------|-----|---------|----------|-----|
| Regresión Lineal | 0.668 | 180.66 | 272.41 | Baseline |
| Lineal + Log Transform | 0.721 | 163.46 | 249.71 | Corrección asimetría |
| **Lasso Polinomial** | **0.907** | **81.85** | **144.40** | **Predicción** ⭐ |
| Backward Elimination | 0.507 | 180.00 | 250.00 | **Inferencia** 🔬 |

### Hallazgos Clave:
- 🌟 **Insolación** es el factor dominante (6x más importante que otros)
- 🌍 Identificados **159 planetas** en zona térmica habitable
- 🔭 **TRAPPIST-1 e** es uno de los candidatos más prometedores (T = 249.7K)

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Scikit-learn** - Modelos de ML (Lasso, PolynomialFeatures, KNN Imputer)
- **Statsmodels** - OLS, backward elimination
- **Matplotlib / Seaborn** - Visualizaciones
- **Jupyter Notebook** - Análisis interactivo

---

## 📊 Visualizaciones

### Distribución de Valores Nulos
![Nulos](images/valores_nulos.png)

### Matriz de Correlación
![Correlación](images/correlacion_df_model.png)

### Real vs Predicho (Lasso Polinomial)
![Predicción](images/real_vs_predicho.png)

---

## 🚀 Cómo Ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/exoplanet-habitability-analysis.git
cd exoplanet-habitability-analysis
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook
```bash
jupyter notebook notebooks/PP1_IA.ipynb
```

### 4. (Opcional) Descargar dataset actualizado
Visita [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) y descarga el archivo PSCompPars más reciente.

---

## 📁 Estructura del Proyecto
```
exoplanet-habitability-analysis/
├── data/                  # Datos originales
├── notebooks/             # Jupyter notebooks
├── images/                # Gráficas generadas
├── docs/                  # Documentación y reporte
├── README.md
└── requirements.txt
```

---

## 🔬 Metodología

1. **Limpieza de datos**: Eliminación de columnas con >50% nulos, imputación KNN
2. **Ingeniería de características**: Transformación logarítmica, eliminación de outliers
3. **Selección de variables**: Análisis de correlación + búsqueda exhaustiva con CV
4. **Modelado**:
   - Regresión lineal múltiple
   - Lasso con características polinomiales (grado 2)
   - Backward elimination para inferencia
5. **Evaluación**: R², MAE, RMSE + validación cruzada

---

## 📚 Referencias

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- Kopparapu et al. (2013). *Habitable Zones Around Main-Sequence Stars*
- Kasting et al. (1993). *Habitable Zones around Main Sequence Stars*

---

## 👨‍💻 Autor

**Juan Angel Candelaria Rodriguez**  
Universidad de Monterrey | Inteligencia Artificial  
📧 juan.candelaria@udem.edu  
🔗 [LinkedIn](https://linkedin.com/in/tu-perfil) | [GitHub](https://github.com/TU_USUARIO)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- NASA Exoplanet Archive por proporcionar datos de acceso público
- Dr. Antonio Martínez Torteya (Universidad de Monterrey)
- Comunidad de astrofísica y ciencia de datos

---

**⭐ Si este proyecto te resultó útil, considera darle una estrella!**
