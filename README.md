# Trabajo 4 - Visualización de datos con Matplotlib y Seaborn

Este proyecto forma parte de una práctica de visualización de datos en Python. El objetivo es analizar un conjunto de datos de ventas minoristas utilizando **Pandas** para preparar la información y **Matplotlib** y **Seaborn** para representarla gráficamente.

He intentado mantener el código sencillo y dividido en funciones para que sea fácil identificar qué parte corresponde a cada tipo de gráfico.

## Dataset

El proyecto utiliza el archivo `superstore_dataset2012.csv` proporcionado en el enunciado. Contiene información sobre pedidos, clientes, productos, ventas, descuentos, beneficios y costes de envío.

## Qué hace el programa

Al ejecutar `visualizacion_ventas.py`:

- carga y comprueba el dataset;
- convierte las fechas al formato adecuado;
- revisa tipos de datos y valores nulos;
- crea gráficos univariantes, bivariantes y multivariantes;
- combina cuatro gráficos en una figura con subplots;
- guarda las visualizaciones generadas en la carpeta `graficos`;
- muestra por consola algunas conclusiones básicas del análisis.

Entre los gráficos incluidos hay un histograma, un boxplot, diagramas de dispersión, una regresión, un gráfico multivariante y un heatmap de correlaciones.

## Estructura del proyecto

```text
trabajo4_visualizacion/
├── visualizacion_ventas.py
├── superstore_dataset2012.csv
├── requirements.txt
├── README.md
├── .gitignore
└── graficos/
    ├── 01_histograma_ventas.png
    ├── 02_boxplot_beneficio_categoria.png
    ├── 03_ventas_vs_beneficio_matplotlib.png
    ├── 04_regresion_ventas_beneficio.png
    ├── 05_multivariante_matplotlib.png
    ├── 06_heatmap_correlaciones.png
    └── 07_resumen_subplots.png
```

## Instalación

Es necesario tener Python instalado. Las bibliotecas utilizadas se pueden instalar con:

```bash
pip install -r requirements.txt
```

También se pueden instalar directamente:

```bash
pip install pandas matplotlib seaborn
```

## Ejecución

El CSV debe estar en la misma carpeta que el archivo Python. Después se puede ejecutar:

```bash
python visualizacion_ventas.py
```

## Algunas conclusiones

Al revisar los datos se observa que las ventas están bastante concentradas en importes bajos, aunque existen operaciones de valor mucho mayor. También aparecen pedidos con beneficio negativo.

La relación entre ventas y beneficio es positiva, pero no perfecta: vender más no significa necesariamente obtener beneficio en todos los pedidos. En estos datos la correlación entre ambas variables es aproximadamente **0,48**.

Por volumen de ventas, **Technology** es la categoría con el total más alto en este conjunto de datos. El heatmap permite comprobar además cómo se relacionan las ventas, el beneficio, los descuentos, la cantidad y el coste de envío.

## Nota

El programa incluye comprobaciones básicas para avisar si falta el CSV, si no se puede leer correctamente o si faltan columnas necesarias para realizar el análisis.
