import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------

ARCHIVO_CSV = "superstore_dataset2012.csv"
CARPETA_GRAFICOS = "graficos"


# ---------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------

def cargar_datos(nombre_archivo):
    """
    Carga el dataset y realiza una preparación básica.
    """

    try:
        df = pd.read_csv(nombre_archivo)
        print("Dataset cargado correctamente.")

    except FileNotFoundError:
        print(f"Error: no se encontró el archivo '{nombre_archivo}'.")
        return None

    except Exception as error:
        print("Se produjo un error al cargar el dataset:")
        print(error)
        return None

    return df


def preparar_datos(df):
    """
    Convierte fechas y variables numéricas al tipo correcto.
    """

    columnas_necesarias = [
        "Sales",
        "Profit",
        "Discount",
        "Quantity",
        "Category",
        "Segment",
        "Order Date",
        "Ship Date"
    ]

    # Comprobamos que las columnas necesarias existen.
    for columna in columnas_necesarias:
        if columna not in df.columns:
            print(f"Error: falta la columna '{columna}' en el dataset.")
            return None

    # Convertimos las fechas.
    # El dataset utiliza formato día/mes/año.
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Convertimos columnas numéricas.
    columnas_numericas = [
        "Sales",
        "Profit",
        "Discount",
        "Quantity"
    ]

    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # Comprobamos si se generaron fechas nulas.
    print("\nFechas no válidas encontradas:")
    print("Order Date:", df["Order Date"].isnull().sum())
    print("Ship Date:", df["Ship Date"].isnull().sum())

    return df


# ---------------------------------------------------------
# EXPLORACIÓN INICIAL
# ---------------------------------------------------------

def explorar_datos(df):
    """
    Muestra información básica sobre el dataset.
    """

    print("\n--- PRIMERAS FILAS ---")
    print(df.head())

    print("\n--- DIMENSIONES DEL DATASET ---")
    print("Filas:", df.shape[0])
    print("Columnas:", df.shape[1])

    print("\n--- TIPOS DE DATOS ---")
    print(df.dtypes)

    print("\n--- VALORES NULOS ---")
    print(df.isnull().sum())

    print("\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
    print(df.describe())


# ---------------------------------------------------------
# GRÁFICOS MATPLOTLIB
# ---------------------------------------------------------

def grafico_histograma(df):
    """
    Histograma de ventas con Matplotlib.
    Permite observar cómo se distribuyen los valores de ventas.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["Sales"].dropna(),
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribución de las ventas")
    plt.xlabel("Ventas")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "01_histograma_ventas.png"
        )
    )

    plt.show()

    # Conclusión:
    # La mayoría de las ventas se concentran en valores relativamente bajos,
    # mientras que existen algunas operaciones con importes mucho mayores.


def grafico_dispersion_matplotlib(df):
    """
    Gráfico de dispersión entre ventas y beneficios.
    """

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["Sales"],
        df["Profit"],
        alpha=0.5
    )

    plt.title("Relación entre ventas y beneficios")
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "03_dispersion_sales_profit_matplotlib.png"
        )
    )

    plt.show()

    # Conclusión:
    # En general, las operaciones con mayores ventas pueden generar mayores
    # beneficios, aunque también existen ventas elevadas con pérdidas.


def grafico_multivariante_matplotlib(df):
    """
    Gráfico multivariante con Matplotlib.
    Relaciona ventas, beneficio y cantidad.
    """

    datos = df[
        ["Sales", "Profit", "Quantity"]
    ].dropna()

    plt.figure(figsize=(8, 5))

    dispersion = plt.scatter(
        datos["Sales"],
        datos["Profit"],
        s=datos["Quantity"] * 15,
        alpha=0.5
    )

    plt.title(
        "Ventas y beneficio según cantidad de productos"
    )
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "05_multivariante_matplotlib.png"
        )
    )

    plt.show()

    # Conclusión:
    # El tamaño de los puntos representa la cantidad de productos vendidos.
    # Esto permite observar simultáneamente ventas, beneficio y cantidad.


# ---------------------------------------------------------
# GRÁFICOS SEABORN
# ---------------------------------------------------------

def grafico_boxplot(df):
    """
    Boxplot de beneficio según categoría.
    """

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Category",
        y="Profit"
    )

    plt.title(
        "Distribución del beneficio por categoría"
    )
    plt.xlabel("Categoría")
    plt.ylabel("Beneficio")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "02_boxplot_profit_categoria.png"
        )
    )

    plt.show()

    # Conclusión:
    # El boxplot permite comparar la distribución de beneficios entre
    # categorías y detectar posibles valores atípicos.


def grafico_regresion(df):
    """
    Gráfico bivariante con línea de regresión.
    """

    datos = df[
        ["Sales", "Profit"]
    ].dropna()

    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=datos,
        x="Sales",
        y="Profit",
        scatter_kws={
            "alpha": 0.4
        }
    )

    plt.title(
        "Relación entre ventas y beneficios"
    )
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "04_regresion_sales_profit.png"
        )
    )

    plt.show()

    # Conclusión:
    # La línea de regresión permite observar una tendencia positiva general
    # entre ventas y beneficio, aunque existe bastante dispersión.


def grafico_heatmap(df):
    """
    Heatmap de correlaciones.
    """

    columnas = [
        "Sales",
        "Profit",
        "Quantity",
        "Discount"
    ]

    datos = df[columnas].dropna()

    correlacion = datos.corr()

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title(
        "Matriz de correlación"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "06_heatmap_correlaciones.png"
        )
    )

    plt.show()

    # Conclusión:
    # El heatmap permite comprobar qué variables numéricas presentan
    # relaciones más fuertes entre sí.


# ---------------------------------------------------------
# SUBPLOTS
# ---------------------------------------------------------

def crear_subplots(df):
    """
    Crea cuatro visualizaciones en una sola figura.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # ---------------------------------
    # 1. Histograma de ventas
    # ---------------------------------

    axes[0, 0].hist(
        df["Sales"].dropna(),
        bins=30,
        edgecolor="black"
    )

    axes[0, 0].set_title(
        "Distribución de ventas"
    )

    axes[0, 0].set_xlabel(
        "Ventas"
    )

    axes[0, 0].set_ylabel(
        "Frecuencia"
    )

    # ---------------------------------
    # 2. Beneficio por categoría
    # ---------------------------------

    sns.boxplot(
        data=df,
        x="Category",
        y="Profit",
        ax=axes[0, 1]
    )

    axes[0, 1].set_title(
        "Beneficio por categoría"
    )

    axes[0, 1].set_xlabel(
        "Categoría"
    )

    axes[0, 1].set_ylabel(
        "Beneficio"
    )

    # ---------------------------------
    # 3. Ventas frente a beneficio
    # ---------------------------------

    axes[1, 0].scatter(
        df["Sales"],
        df["Profit"],
        alpha=0.4
    )

    axes[1, 0].set_title(
        "Ventas frente a beneficio"
    )

    axes[1, 0].set_xlabel(
        "Ventas"
    )

    axes[1, 0].set_ylabel(
        "Beneficio"
    )

    # ---------------------------------
    # 4. Ventas totales por categoría
    # ---------------------------------

    ventas_categoria = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    axes[1, 1].bar(
        ventas_categoria.index,
        ventas_categoria.values
    )

    axes[1, 1].set_title(
        "Ventas totales por categoría"
    )

    axes[1, 1].set_xlabel(
        "Categoría"
    )

    axes[1, 1].set_ylabel(
        "Ventas totales"
    )

    # Título general.
    fig.suptitle(
        "Resumen del análisis de ventas de Superstore",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    plt.savefig(
        os.path.join(
            CARPETA_GRAFICOS,
            "07_resumen_subplots.png"
        )
    )

    plt.show()

    # Conclusión:
    # Esta figura resume distintos aspectos del dataset:
    # distribución de ventas, beneficios por categoría,
    # relación ventas-beneficio y ventas totales por categoría.


# ---------------------------------------------------------
# CONCLUSIONES GENERALES
# ---------------------------------------------------------

def mostrar_conclusiones(df):
    """
    Calcula algunas conclusiones básicas del análisis.
    """

    correlacion = df[
        ["Sales", "Profit"]
    ].corr().iloc[0, 1]

    porcentaje_perdidas = (
        (df["Profit"] < 0).mean() * 100
    )

    ventas_categoria = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print("\n--- CONCLUSIONES GENERALES ---")

    print(
        f"Correlación entre ventas y beneficio: "
        f"{correlacion:.2f}"
    )

    print(
        f"Porcentaje de operaciones con pérdidas: "
        f"{porcentaje_perdidas:.2f}%"
    )

    if not ventas_categoria.empty:

        print(
            "Categoría con mayores ventas:",
            ventas_categoria.index[0]
        )


# ---------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------

def main():

    # Creamos la carpeta donde se guardarán los gráficos.
    os.makedirs(
        CARPETA_GRAFICOS,
        exist_ok=True
    )

    # Cargamos los datos.
    df = cargar_datos(
        ARCHIVO_CSV
    )

    if df is None:
        return

    # Preparamos los datos.
    df = preparar_datos(
        df
    )

    if df is None:
        return

    # Exploración inicial.
    explorar_datos(
        df
    )

    # Gráficos Matplotlib.
    grafico_histograma(
        df
    )

    grafico_dispersion_matplotlib(
        df
    )

    grafico_multivariante_matplotlib(
        df
    )

    # Gráficos Seaborn.
    grafico_boxplot(
        df
    )

    grafico_regresion(
        df
    )

    grafico_heatmap(
        df
    )

    # Figura con cuatro subplots.
    crear_subplots(
        df
    )

    # Conclusiones.
    mostrar_conclusiones(
        df
    )

    print(
        "\nAnálisis finalizado correctamente."
    )

    print(
        f"Los gráficos se han guardado en la carpeta "
        f"'{CARPETA_GRAFICOS}'."
    )


# Ejecutamos el programa.
if __name__ == "__main__":
    main()
