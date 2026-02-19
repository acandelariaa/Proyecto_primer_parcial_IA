# Inteligencia Artiificial - Proyecto parcial 1
### Juan Angel Candelaria Rodriguez 653728

Este proyecto tiene como finalidad predecir la temperatura de equilibrio de un planeta en referencia a la tierra para determinar si puede ser potencialmente habitable.

#Predicción de la Temperatura de Equilibrio de Exoplanetas
## ¿Qué tan parecido a la Tierra puede ser un planeta fuera de nuestro sistema solar?

**Inteligencia Artificial | Universidad de Monterrey**  
**Proyecto parcial – Unidad 1: Modelos de Regresión**

---


### Contexto


Este enfoque de predicción de habitabilidad de un exoplaneta tiene una justificación un poco mas personal, ya que siempre me ha gustado mucho el espacio y todo lo relacionado con las estrellas, la vida en otros planetas, etc.

Partiendo de esto, siempre ha sido una curiosidad intrinseca de la humanidad el saber que hay mas afuera, conforme ha avanzado la tecnologia, cada vez hemos sido mas capaces de buscar cuerpos celestes, y catalogarlos en comparación a nuestro planeta tierra. Asi mismo, una de las principales preguntas de la comunidad cientifica, ha sido que planeta es potencialmente habitable para nosotros los humanos, esto debido a factores como el deterioro del medio ambiente, sobrepoblacion, entre otras cosas. La busqueda de otro planeta que pueda albergar vida, esta a expensas de lo similar que sea ese planeta a la tierra, ya que nos hemos desarrollado en las condiciones que este provee, de modo que un enfoque seria buscar un lugar similar en el extenso universo conocido.

La búsqueda de un mundo similar al nuestro no es arbitraria — nos hemos desarrollado bajo condiciones muy específicas, de modo que el punto de partida más natural es buscar planetas que compartan esas mismas condiciones. Entre todos los factores que definen si un planeta puede albergar vida, la temperatura de equilibrio destaca como un indicador integrador: refleja directamente la cantidad de energía que recibe un planeta de su estrella, y está físicamente determinada por características que sí podemos observar a distancia. Por ello, prestaremos especial atención a las variables `st_teff` (temperatura efectiva de la estrella), `st_rad` (radio estelar), `pl_orbsmax` (semieje mayor orbital), `pl_insol` (insolación recibida) y `pl_orbeccen` (excentricidad orbital), entre otros; ya que son los parámetros que, tanto físicamente como en la literatura científica, gobiernan directamente el valor de `pl_eqt`.

La temperatura de equilibrio de la **Tierra es aproximadamente 255 K** (-18°C).

Los planetas cuya temperatura de equilibrio se encuentra en el rango de ~200 K a 320 K se consideran térmicamente compatibles con la existencia de agua líquida y, por tanto, candidatos a ser habitables. Obviamente también hay más variables que pueden influir, como la masa del planeta, si es rocoso, si está en la zona habitable con respecto a su estrella más cercana, entre otros.

Sin embargo, de modo que estos factores que pueden determinar la habitabilidad estan relacionados entre si, de modo que la temperatura seria un buen enfoque para este contexto.

### Pregunta de Investigación

> *¿Qué características de un planeta permiten predecir qué tan similar es la temperatura de equilibrio de un exoplaneta a la de la Tierra, y qué planetas se encuentran dentro del rango compatible con la habitabilidad?*

### Variable de salida

Para hacer la predicción interpretable en términos terrestres, utilizaremos una variable llamada **Temperatura de Equilibrio (pl_eqt)**, el cual estara medido en Kelvin:

`pl_eqt`

De modo que este indice nos podria decir lo siguiente:
- Un valor cercano a 255 K indica que el planeta tiene una temperatura de equilibrio que la Tierra.
- Un valor < 255 K indica un planeta más frío que la Tierra.
- Un valor de > 255 K indica un planeta más caliente que la Tierra.


### Justificación



La elección de un enfoque de regresión tiene sentido por varias razones. Primero, pl_eqt es una variable continua y cuantitativa, no una categoría de sí o no. Además, existe una relación física ya conocida entre las características de la estrella, la órbita y la temperatura del planeta. La idea no es etiquetar planetas como habitables o no habitables, sino cuantificar qué tan cerca está cada uno del perfil térmico de la Tierra, que es esencialmente un problema de predicción.


--- 

### Fuentes

Los datos fueron recopilados del **NASA Exoplanet Archive** (https://exoplanetarchive.ipac.caltech.edu), el repositorio oficial de la NASA para datos de exoplanetas confirmados. El archivo utilizado es el **Planetary Systems Composite Parameters** (PSCompPars), descargado de la pagina oficial de la NASA.

Este dataset esta conformado por multiples mediciones para cada planeta avistado, observaciones de múltiples telescopios y misiones espaciales, incluyendo Kepler, K2, TESS y observatorios terrestres.




---

### Siguiente pagina

[>>> Exploracion de datos](explorar_datos.md)
