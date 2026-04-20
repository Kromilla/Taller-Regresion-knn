<div align="center">
  <h1>Taller Práctico de Machine Learning</h1>
  <p><b>Análisis y Predicción de Datos mediante Regresión Lineal y KNN</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org)
</div>

<br>

Este repositorio fue creado como entrega académica para un taller demostrativo enfocado en los fundamentos matemáticos de Machine Learning. El objetivo central no es sólo usar librerías, sino **construir algoritmos predictivos desde cero**, contrastándolos con frameworks modernos de producción.

## Características Principales

1. **Regresión Lineal Aplicada (Boston Housing Dataset):**
   * **Análisis Exploratorio (EDA):** Evaluación de correlaciones visuales frente a precios de vivienda (`MEDV` vs `LSTAT`, `RM`).
   * **Implementación Matemática:** Algoritmo manual de **Gradiente Descendente** (MSE, Learning Rates, Iteraciones).
   * **Resolución Analítica:** Implementación puramente algebraica utilizando la **Ecuación Normal**.
   * **Implementación Industrial:** Uso optimizado respaldado por `sklearn`.

2. **K-Nearest Neighbors (Dataset Iris):**
   * **KNN Desde Cero:** Implementación propia de funciones paramétricas con distancias (Votación de mayoría).
   * **Experimento de Distancias:** Comparativa directa calculando distancia *Euclidiana* vs distancia *Manhattan*.
   * **Análisis de Overfitting:** Alteración del factor `K` revelando puntos críticos de quiebre (memorización de conjunto de prueba).
   * **Impacto por Normalización:** Desempeños con estandarización *Z-Score* y de límites *Min-Max*.

---

## Estructura del Repositorio

* `Taller_ML_Solucion_Paso_A_Paso.ipynb` — Documento iterativo Jupyter detallando línea a línea todo el proceso y las conclusiones por gráfica.
* `Taller_Respuestas_Final.md` — Recopilatorio resuelto, debidamente respondido y exportado de las preguntas generadas durante el desarrollo del taller.
* `HousingData.csv` — Dataset Boston modificado estructuralmente (incluye `NaN`).
* `Taller de Regresión Lineal y Knn.docx` — Base de las reglas de construcción de la práctica.

---

## Conclusiones Destacadas

* Se logró una convergencia perfecta con **Gradiente Descendente**, logrando un $MSE = 21.91$.
* El algoritmo optimizado subyacente de **Scikit-Learn** en C/C++ sigue demostrando supremacía en coste temporal promedio pero la Ecuación Normal exhibió solidez.
* En distancias matemáticas, al entrenar **KNN** en set florales controlados, hubo un empate de perfección estadística frente a distancias Euclidiana y Manhattan.

---

## Instalación y Uso Rápido

Para correr este entorno localmente y visualizar los gráficos, debes instalar las dependencias básicas:

```bash
# 1. Clona el repositorio
git clone https://github.com/Kromilla/Taller-Regresion-knn.git
cd Taller-Regresion-knn

# 2. Instala los requerimientos
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Abre el archivo principal en Jupyter
jupyter notebook Taller_ML_Solucion_Paso_A_Paso.ipynb
```

> Taller práctico enfocado en el desarrollo académico y la asimilación de algoritmos de Machine Learning.
