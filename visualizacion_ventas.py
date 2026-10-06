from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "superstore_dataset2012.csv"
OUTPUT_DIR = BASE_DIR / "graficos"

# Columnas que vamos a utilizar en el análisis
COLUMNAS_NECESARIAS = {
    "Order Date",
    "Ship Date",
    "Category",
    "Segment",
    "Sales",
    "Profit",
    "Quantity",
    "Discount",
    "Shipping Cost",
}


def cargar_datos():
    """Carga el CSV, comprueba sus columnas y prepara fechas y datos numéricos."""
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{DATASET.name}'. "
            "Debe estar en la misma carpeta que este programa."
        )

    try:
        datos = pd.read_csv(DATASET)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"No se pudo leer correctamente el CSV: {error}") from error

    columnas_faltantes = COLUMNAS_NECESARIAS - set(datos.columns)
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: "
            + ", ".join(sorted(columnas_faltantes))
        )

    # Las fechas del archivo están guardadas como texto. Las convertimos a datetime.
    datos["Order Date"] = pd.to_datetime(
        datos["Order Date"], format="%m/%d/%Y", errors="coerce"
    )
    datos["Ship Date"] = pd.to_datetime(
        datos["Ship Date"], format="%m/%d/%Y", errors="coerce"
    )

    # Nos aseguramos de que las variables usadas en cálculos sean numéricas.
    columnas_numericas = ["Sales", "Profit", "Quantity", "Discount", "Shipping Cost"]
    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    return datos


def explorar_datos(datos):
    """Muestra información básica para conocer la estructura y calidad del dataset."""
    print("\n--- EXPLORACIÓN INICIAL ---")
    print(f"Filas: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    print("\nPrimeras 5 filas:")
    print(datos.head())

    print("\nTipos de datos:")
    print(datos.dtypes)

    print("\nValores nulos por columna:")
    print(datos.isnull().sum())

    print("\nResumen de variables numéricas:")
    print(datos[["Sales", "Profit", "Quantity", "Discount", "Shipping Cost"]].describe())


def grafico_univariante_matplotlib(datos):
    """Histograma de ventas realizado únicamente con Matplotlib."""
    plt.figure(figsize=(9, 5))
    plt.hist(datos["Sales"].dropna(), bins=30, edgecolor="black", alpha=0.75)
    plt.title("Distribución de las ventas")
    plt.xlabel("Ventas")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_histograma_ventas.png", dpi=150)

    # Conclusión: la distribución de Sales está concentrada en importes bajos y
    # presenta algunos valores de venta mucho más altos que la mayoría.


def grafico_univariante_seaborn(datos):
    """Boxplot de beneficios por categoría realizado con Seaborn."""
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=datos,
        x="Category",
        y="Profit",
        hue="Category",
        palette="Set2",
        legend=False,
    )
    plt.title("Distribución del beneficio por categoría")
    plt.xlabel("Categoría")
    plt.ylabel("Beneficio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_boxplot_beneficio_categoria.png", dpi=150)

    # Conclusión: existen operaciones con pérdidas y también valores extremos de
    # beneficio. El boxplot permite comparar fácilmente la dispersión entre categorías.


def grafico_bivariante_matplotlib(datos):
    """Relación entre ventas y beneficios mediante un scatter de Matplotlib."""
    plt.figure(figsize=(9, 5))
    plt.scatter(datos["Sales"], datos["Profit"], alpha=0.45)
    plt.title("Relación entre ventas y beneficios")
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")
    plt.axhline(0, linewidth=1, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_ventas_vs_beneficio_matplotlib.png", dpi=150)

    # Conclusión: ventas altas no garantizan siempre beneficios positivos. En general
    # existe una relación positiva moderada, pero aparecen operaciones con pérdidas.


def grafico_bivariante_seaborn(datos):
    """Regresión entre ventas y beneficios mediante Seaborn."""
    muestra = datos[["Sales", "Profit"]].dropna()

    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=muestra,
        x="Sales",
        y="Profit",
        scatter_kws={"alpha": 0.35},
        line_kws={"linewidth": 2},
    )
    plt.title("Ventas y beneficios con línea de regresión")
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_regresion_ventas_beneficio.png", dpi=150)

    # Conclusión: la línea de regresión resume la tendencia positiva entre Sales y
    # Profit, aunque la dispersión indica que otros factores también influyen.


def grafico_multivariante_matplotlib(datos):
    """Scatter multivariante: ventas, beneficio y descuento con Matplotlib."""
    muestra = datos[["Sales", "Profit", "Discount"]].dropna()

    plt.figure(figsize=(9, 5))
    puntos = plt.scatter(
        muestra["Sales"],
        muestra["Profit"],
        c=muestra["Discount"],
        cmap="viridis",
        alpha=0.55,
    )
    plt.colorbar(puntos, label="Descuento")
    plt.title("Ventas, beneficios y nivel de descuento")
    plt.xlabel("Ventas")
    plt.ylabel("Beneficio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_multivariante_matplotlib.png", dpi=150)

    # Conclusión: al añadir Discount como tercera variable se puede observar si los
    # niveles de descuento están relacionados con zonas de mayor o menor beneficio.


def grafico_multivariante_seaborn(datos):
    """Heatmap de correlaciones entre variables numéricas usando Seaborn."""
    columnas = ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
    correlaciones = datos[columnas].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlaciones,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("Correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_heatmap_correlaciones.png", dpi=150)

    # Conclusión: el heatmap permite comparar simultáneamente varias relaciones.
    # En este dataset Sales y Profit presentan una correlación positiva moderada.


def figura_subplots(datos):
    """Crea una figura 2x2 con cuatro visualizaciones distintas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Matplotlib: histograma de ventas
    axes[0, 0].hist(datos["Sales"].dropna(), bins=30, edgecolor="black", alpha=0.75)
    axes[0, 0].set_title("Distribución de ventas")
    axes[0, 0].set_xlabel("Ventas")
    axes[0, 0].set_ylabel("Frecuencia")

    # 2. Seaborn: boxplot de beneficio por categoría
    sns.boxplot(
        data=datos,
        x="Category",
        y="Profit",
        hue="Category",
        palette="Set2",
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Beneficio por categoría")
    axes[0, 1].set_xlabel("Categoría")
    axes[0, 1].set_ylabel("Beneficio")

    # 3. Matplotlib: ventas frente a beneficio
    axes[1, 0].scatter(datos["Sales"], datos["Profit"], alpha=0.35)
    axes[1, 0].axhline(0, linewidth=1, linestyle="--")
    axes[1, 0].set_title("Ventas frente a beneficio")
    axes[1, 0].set_xlabel("Ventas")
    axes[1, 0].set_ylabel("Beneficio")

    # 4. Seaborn: ventas totales por categoría
    sns.barplot(
        data=datos,
        x="Category",
        y="Sales",
        estimator="sum",
        errorbar=None,
        hue="Category",
        palette="Set2",
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Ventas totales por categoría")
    axes[1, 1].set_xlabel("Categoría")
    axes[1, 1].set_ylabel("Ventas totales")

    fig.suptitle("Análisis visual de ventas minoristas - Superstore 2012", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "07_resumen_subplots.png", dpi=150)

    # Conclusión general: esta figura permite comparar en una sola vista la
    # distribución de ventas, la rentabilidad por categoría, la relación entre
    # ventas y beneficio y el volumen total vendido por categoría.


def mostrar_conclusiones(datos):
    """Imprime algunos resultados numéricos que respaldan la lectura de los gráficos."""
    correlacion = datos[["Sales", "Profit"]].corr().loc["Sales", "Profit"]
    ventas_categoria = datos.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    porcentaje_perdidas = (datos["Profit"] < 0).mean() * 100

    print("\n--- CONCLUSIONES RESUMIDAS ---")
    print(f"Correlación Sales-Profit: {correlacion:.2f}")
    print(f"Operaciones con Profit negativo: {porcentaje_perdidas:.1f}%")
    print("\nVentas totales por categoría:")
    print(ventas_categoria.round(2))


def main():
    try:
        datos = cargar_datos()
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Estilo general para mejorar la apariencia de los gráficos.
        sns.set_theme(style="whitegrid")

        explorar_datos(datos)
        grafico_univariante_matplotlib(datos)
        grafico_univariante_seaborn(datos)
        grafico_bivariante_matplotlib(datos)
        grafico_bivariante_seaborn(datos)
        grafico_multivariante_matplotlib(datos)
        grafico_multivariante_seaborn(datos)
        figura_subplots(datos)
        mostrar_conclusiones(datos)

        print(f"\nGráficos guardados correctamente en: {OUTPUT_DIR}")
        plt.show()

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
