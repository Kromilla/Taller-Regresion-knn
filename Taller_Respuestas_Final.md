# Taller de Regresión Lineal y Knn - Respuestas

A continuación se presentan las respuestas del análisis de los modelos aplicados en el dataset de Boston Housing y el dataset de Iris (KNN).

---

## Parte 1: Análisis Exploratorio (EDA)

**¿Qué variable parece tener mayor relación con el precio?**  
Las variables que presentan la correlación más fuerte con el precio (`MEDV`) son el **`RM` (Número de habitaciones)** con una relación positiva clara, y **`LSTAT` (Porcentaje de estatus social bajo)** que mantiene una relación inversamente proporcional (negativa).

**¿Existen relaciones lineales claras?**  
Sí. La relación lineal más limpia a simple vista es entre `MEDV` y `RM`, formando una clara tendencia lineal ascendente (a mayor cantidad de cuartos, el valor mediano sube notoriamente).

---

## Parte 2: Regresión Lineal desde Cero (Gradiente Descendente)

**¿El error converge?**  
Sí, al observar la gráfica de evolución del error iterativo del algoritmo de gradiente descendente, el costo cuadrático cae en picada y después de unas pocas docenas de iteraciones logra estabilizarse en su valor mínimo, mostrando un **MSE total convergido de 21.91**.

**¿Cómo afecta el learning rate (alpha)?**  
El *learning rate* determina la longitud del "salto" al minimizar el error. Si el paso es muy pronunciado/alto, el modelo nunca convergería y arrojaría un error matemático de divergencia; si fuera excesivamente suave/bajo, tomaría cientos de miles de iteraciones llegar a la estabilización computacionalmente. Al normalizar los datos desde un inicio, la tasa configurada (`0.1`) logró el balance perfecto para que la convergencia fuese veloz y exacta.

---

## Parte 3: Solución con Ecuación Normal

**¿Los resultados son similares?**  
Son prácticamente idénticos (el MSE de ambos métodos evaluados en el notebook resultó exactamente de **21.91**).

**¿Qué ventajas/desventajas observas?**  
- **Desventaja Fundamental:** La ecuación normal necesita realizar una inversión de matriz (complejidad de $O(d^3)$). Si se aplicara al dataset gigantescos, el proceso matemático desbordaría el equipo.
- **Múltiple Ventaja:** Al resolverlo matemáticamente, nunca es necesario proveer un "Learning rate" (Alpha variable) ni iterar un loop, alcanza la respuesta analítica en una solitaria ejecución sin lidiar con calibraciones artificiales.

---

## Parte 5: Comparación de Métodos

| Método | MSE | Tiempo Promedio (s) | Complejidad |
| :--- | :--- | :--- | :--- |
| **Gradiente Descendente** | 21.91 | ~ 0.020 | Iterativa: Alta carga de ciclos iterativos pero ágil con matrices grandes. |
| **Ecuación Normal** | 21.91 | ~ 0.0002 | $O(d^3)$: Complejidad analítica, Inversión matricial paralizante con big data. |
| **sklearn (LinearRegression)**| 21.91 | ~ 0.003 | Altamente optimizado (Librería probada, Cython/C++, sin fallas obvias). |

**¿Cuál método es más eficiente?**  
`sklearn`. Aunque la Ecuación Normal teórica logró ser procesada con rapidez, localmente *Scikit-Learn* sub-categoriza la descomposición interna evitando saturaciones de memoria, siendo sumamente más eficiente sobre condiciones reales de carga.

**¿Cuál es más escalable?**  
El **Gradiente Descendente**. Su diseño iterativo permite que se construya un modelo dividiendo datos masivos y cargándolos al momento del procesamiento; algo esencial si manejásemos millones de registros que harían colapsar ecuaciones de matriz.

**¿Cuál usarías en producción?**  
Definitivamente **Sklearn (`LinearRegression`)**. Brinda algoritmos súper eficientes y seguros para entornos de productibilidad, previene caídas técnicas y asiste el código local delegando optimizaciones lógicas escritas en C. 

---

## Parte 6: Interpretación del Modelo

**¿Qué significa cada uno (coeficientes obtenidos)?**  
Los coeficientes demuestran la *tasa de alteración real* en la predicción.
Ejemplo general desde el dataset de Boston de los Coeficientes: 
- El coeficiente de `"RM"` (Habitaciones) indicó un peso superior a **+4.01**. Por ende, con cada habitación extra construída, el costo hipotético de la residencia asciende considerablemente (+$4,010 apprx). 
- El coeficiente atribuido al aire no limpio (`NOX`, índice tóxico de nitrógeno) fue de **-18.15**, dictando un desplome bestial y restrictivo para la valoración final predicha sobre la casa.

**¿Qué variable impacta más el precio?**  
Las toxinas químicas/contaminación ambiental (`NOX` -18.15) reducen el valor radicalmente de manera directa, así como las habitaciones habitables lo fomentan en positivo velozmente (`RM` 4.01).

**¿El modelo es confiable?**  
El **$R^2$ Score fue de 0.7404**, definiendo que se puede fiar del mismo un 74% bajo explicaciones y variaciones del costo nativo frente al pronóstico de precios. En este plano académico es enormemente confiable, para finanzas empresariales se priorizarían modelos que asfixien los Outliers atípicos.

---

## Parte B: Experimentos Avanzados (KNN con el dataset Iris)

**Experimento 1: ¿Dónde ocurre overfitting (sobreajuste)?**  
Ocurre en **K = 1**. Al correr el algoritmo bajo una vecindad singular (un sólo voto vecino), evaluando el training contra sí mismo, entregó precisión artificial de `1.00` (100%). Literalmente "memorizó" los registros pasados pero no es capaz de trazar fronteras de generalización realistas y colapsaría con simples anomalías futuras.

**Experimento 2: ¿Cómo cambia la precisión al normalizar?**  
A nivel KNN (en `Iris` exclusivamente) no cambió el indicador casi en absoluto manteniendo métricas sobresalientes (como se aprecia). Debido puramente a que las métricas biológicas (pétalos / longitud sépalos) se comportaron unificados bajo márgenes de un sólo dígito por igual (de entre 0cm a 7cm). Ante contextos de distintas escalas drásticas (ej: años de antigüedad en una métrica y peso en miles de kilogramos en otra métrica simultanea) saltaría la urgente necesidad de implementar un *Scaler* normalizador de inmediado para salvar al modelo probabilístico.

**Experimento 3: ¿Cuál funciona mejor y por qué (Distancia Euclidiana vs Manhattan)?**  
Ambos procesaron calificaciones de top-tier perfectas en *tests exactos* limitados del set floral de Fisher; debido a ser datos contiguos perfectos para algoritmos elementales. En estándar de ciencia de datos, usaríamos la **Euclidiana** por defecto porque es invariante frente a rotaciones, encontrando un vector del "camino más corto en línea recta" (lógico para dimensiones de la vida real en el espacio), recomendando a la paralela *Manhattan* sí, en el futuro procesásemos información encuadrada sobre datos binarios estriados (ciudades, cuadrículas y grafos urbanos) donde "cortar esquinas" no es un reflejo de viabilidad posible al desplazarse en forma cruz.
