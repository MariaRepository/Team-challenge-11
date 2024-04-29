import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr






### Funcion: plot_features_num_regression (María)

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):

    """
    Descripción: realiza análisis de regresión lineal entre una columna objetivo y las columnas numéricas de un DataFrame,
    filtrando aquellas que tienen una correlación significativa y un valor p bajo.

    Argumentos:
    df (DataFrame): El DataFrame que contiene los datos.
    target_col (str): El nombre de la columna objetivo que se usará en el análisis de regresión.
    columns (list): Una lista de nombres de columnas a considerar. Si está vacía, se considerarán todas las columnas numéricas del DataFrame.
    umbral_corr (float): El umbral de correlación mínimo para considerar una columna como relevante en el análisis.
    pvalue (float): El valor p máximo para considerar una correlación como estadísticamente significativa.

    Retorna:
    list: pairplot del dataframe considerando la columna designada por "target_col" y aquellas incluidas en "column" que cumplan los requisitos indicados.
    """
    # Comprobar si la lista de columnas está vacía
    if not columns:
        # Obtener todas las columnas numéricas del DataFrame
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Verificar si el target_col no está vacío
    if not target_col:
        raise ValueError("El argumento 'target_col' no puede estar vacío.")
    
    # Verificar si el target_col es una columna numérica
    if target_col not in df.select_dtypes(include=['float64', 'int64']).columns:
        raise ValueError("El argumento 'target_col' debe ser una columna numérica.")
        
    # Filtrar columnas basadas en umbral_corr y pvalue
    filtered_columns = []
    for col in columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[target_col], df[col])
            if abs(correlation) > umbral_corr and (pvalue is None or p_value < pvalue):
                filtered_columns.append(col)
            else:
                print(f"La columna numérica '{col}' tiene una correlación absoluta menor o igual que el umbral ({umbral_corr}).")
                if pvalue is not None and p_value >= pvalue:
                    print(f"Además, el valor p ({p_value}) es mayor o igual que el valor especificado ({pvalue}).")
        
    # Dividir las columnas en grupos de máximo cinco para pairplot
    for i in range(0, len(filtered_columns), 5):
        sns.pairplot(df[filtered_columns[i:i+5] + [target_col]], kind='reg', diag_kind='kde')
        plt.show()
    
    return filtered_columns
# Ejemplo de uso
plot_features_num_regression(df_inmo, target_col="median_house_value", columns=[], umbral_corr=0.5, pvalue=0.05)




# Funcion: plot_features_cat_regression (Fernando)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway


def get_features_cat_regression(dataframe, target_col, pvalue=0.05):
    """
    Selección de características categóricas significativas para regresión.

    Argumentos:
    dataframe (DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna que es el objetivo de la regresión.
    pvalue (float): Nivel de significancia para el test estadístico. Por defecto 0.05.

    Retorna:
    list: Lista de características categóricas significativas.
    """
    
    # Comprobación de la existencia de la columna target_col
    if target_col not in dataframe.columns:
        print(f"Error: La columna {target_col} no existe en el DataFrame.")
        return None
    
    # Comprobación de que target_col sea numérica
    if not np.issubdtype(dataframe[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es numérica.")
        return None
    
    # Comprobación de pvalue válido
    if not isinstance(pvalue, float) or pvalue <= 0 or pvalue >= 1:
        print("Error: pvalue debe ser un valor float en el rango (0, 1).")
        return None
    
    # Obtención de columnas categóricas
    cat_columns = dataframe.select_dtypes(include=['object', 'category']).columns
    
    # Comprobación de existencia de columnas categóricas
    if len(cat_columns) == 0:
        print("Error: No se encontraron columnas categóricas en el DataFrame.")
        return None
    
    # Comprobación de la relación entre cada columna categórica y target_col
    significant_features = []
    for col in cat_columns:
        contingency_table = pd.crosstab(dataframe[col], dataframe[target_col])
        if chi2_contingency(contingency_table)[1] < pvalue:
            significant_features.append(col)
    
    return significant_features


def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Grafica histogramas agrupados para características categóricas en relación con target_col.

    Argumentos:
    dataframe (DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna que es el objetivo de la regresión. Por defecto "".
    columns (list): Lista de nombres de columnas categóricas a considerar. Por defecto [].
    pvalue (float): Nivel de significancia para el test estadístico. Por defecto 0.05.
    with_individual_plot (bool): Si True, muestra histogramas individuales para cada característica. Por defecto False.

    Retorna:
    list: Lista de características categóricas significativas seleccionadas.
    """

    # Verificación de entrada para target_col y columns
    if not isinstance(target_col, str) or not isinstance(columns, list):
        print("Error: target_col debe ser un string y columns debe ser una lista.")
        return None

    # Verificación de existencia de target_col en el DataFrame
    if target_col not in dataframe.columns:
        print(f"Error: La columna {target_col} no existe en el DataFrame.")
        return None
    
    # Verificación de tipo de pvalue
    if not isinstance(pvalue, float) or pvalue <= 0 or pvalue >= 1:
        print("Error: pvalue debe ser un valor float en el rango (0, 1).")
        return None
    
    # Si no se proporciona ninguna columna, se seleccionan las columnas categóricas automáticamente
    if len(columns) == 0:
        columns = list(dataframe.select_dtypes(include=['object', 'category']).columns)
    
    # Obtención de características categóricas significativas
    significant_features = get_features_cat_regression(dataframe, target_col, pvalue)
    if significant_features is None:
        return None
    
    # Verificación de la existencia de características significativas
    if len(significant_features) == 0:
        print("No se encontraron características categóricas significativas.")
        return None
    
    # Plot de histogramas agrupados para características categóricas significativas
    for feature in significant_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=dataframe, x=feature, hue=target_col, multiple="stack")
        plt.title(f"Histograma agrupado para {feature} en relación con {target_col}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        plt.legend(title=target_col)
        plt.show()

        # Plot de histogramas individuales si se especifica
        if with_individual_plot:
            for value in dataframe[feature].unique():
                plt.figure(figsize=(6, 4))
                sns.histplot(data=dataframe[dataframe[feature] == value], x=target_col)
                plt.title(f"Histograma de {target_col} para {feature}={value}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()
    
    return significant_features

