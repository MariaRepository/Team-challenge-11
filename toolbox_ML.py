# *********************************************
### FUNCIONES CONTENIDAS Y EJEMPLO DE USO ###
# *********************************************

    # describe_df -> describe_df(dataframe)

    # tipifica_variables -> tipifica_variables(dataframe, umbral_categoria, umbral_continua)

    # get_features_num_regression -> get_features_num_regression(df, target_col, umbral_corr, pvalue=None)

    # plot_features_num_regression -> plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None)

    # get_features_cat_regression -> get_features_cat_regression(dataframe, target_col, pvalue=0.05)

    # plot_features_cat_regression -> plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False)

    # eval_model -> eval_model(target, predictions, problem_type, metrics)
        # Ejemplos de uso:
            # eval_model(y_true, y_pred, 'regression', ['RMSE', 'MAE', 'GRAPH'])
            # eval_model(y_true, y_pred, 'classification', ['ACCURACY', 'PRECISION', 'RECALL', 'MATRIX', 'PRECISION_1', 'RECALL_2'])

    # get_features_num_classification -> get_features_num_classification(df, target_col, p_value= 0.05)

    # plot_features_num_classification -> plot_features_num_classification(df, target_col="target", columns=[], pvalue=0.05)

    # get_features_cat_classification -> get_features_cat_classification(df, target_col, normalize=False, mi_threshold=0.0)

    # plot_features_cat_classification -> plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False)

    # super_selector -> super_selector(dataset, target_col="", selectores=None, hard_voting=[])
        #Ejemplo: 
            # selectores = {
                # "KBest": 5,
                # "FromModel": [RandomForestClassifier(), 5],
                # "RFE": [LogisticRegression(), 5, 1],
                # "SFS": [RandomForestClassifier(), 5] }

#############################################################################################################################

# ****************************************************
### BIBLIOTECAS ###
# ****************************************************

# Para el manejo y análisis de datos
import pandas as pd
import numpy as np
from collections import Counter

# Para la visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Para el análisis estadístico
from scipy.stats import pearsonr, chi2_contingency, f_oneway

# Para la evaluación de modelos
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Para la preparación de datos y modelos
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Selección de características
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, mutual_info_classif, SequentialFeatureSelector, f_regression
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# Modelos de regresión
from sklearn.linear_model import LinearRegression, LogisticRegression

# Modelos de ensamble
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Para el cálculo de la información mutua
from sklearn.metrics import mutual_info_score

# Para el uso de CatBoost
from catboost import CatBoostClassifier, Pool

# Para ignorar advertencias
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import NotFittedError


# ************************************************************************
### FUNCIONES ###
# ************************************************************************

### Funcion: describe_df 

def describe_df(dataframe):
    """
    Esta función analiza:
    - Los tipos de datos
    - Los valores faltantes
    - Los valores únicos
    - La cardinalidad
    de las variables de un DataFrame. 

    Argumentos:
    dataframe: DataFrame de Pandas

    Retorna:
    DataFrame donde las filas representan los tipos de datos, valores faltantes, etc.,
    y las columnas representan las variables del DataFrame.
    """

    # Lista de columnas del DataFrame
    lista_columnas = dataframe.columns.tolist()
    
    # Diccionario para almacenar los parámetros de cada columna
    diccionario_parametros = {}
    
    # Iteración sobre las columnas del DataFrame
    for columna in lista_columnas:
        # Tipo de datos de la columna
        tipo = dataframe[columna].dtype
        
        # Porcentaje de valores faltantes en la columna
        porc_nulos = round(dataframe[columna].isna().sum() / len(dataframe) * 100, 2)
        
        # Valores únicos en la columna
        valores_no_nulos = dataframe[columna].dropna()
        unicos = valores_no_nulos.nunique()
        
        # Cardinalidad de la columna
        cardinalidad = round(unicos / len(valores_no_nulos) * 100, 2)
        
        # Almacenar los parámetros de la columna en el diccionario
        diccionario_parametros[columna] = [tipo, porc_nulos, unicos, cardinalidad]
    
    # Construcción del DataFrame de resumen
    df_resumen = pd.DataFrame(diccionario_parametros, index=["DATE_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"])
    
    # Retorno del DataFrame de resumen
    return df_resumen

#####################################################################################################################

### Funcion: tipifica_variables 

def tipifica_variables(dataframe, umbral_categoria, umbral_continua):
    """
    Esta función sugiere el tipo de variable para cada columna de un DataFrame
    basándose en la cardinalidad y umbrales proporcionados.

    Argumentos:
    dataframe: DataFrame de Pandas
    umbral_categoria: Entero, umbral para la cardinalidad que indica cuándo considerar una variable categórica.
    umbral_continua: Flotante, umbral para el porcentaje de cardinalidad que indica cuándo considerar una variable numérica continua.

    Retorna:
    DataFrame con dos columnas: "nombre_variable" y "tipo_sugerido",
    donde cada fila contiene el nombre de una columna del DataFrame y una sugerencia del tipo de variable.
    """

    # Lista para almacenar las sugerencias de tipos de variables
    sugerencias_tipos = []

    # Iteración sobre las columnas del DataFrame
    for columna in dataframe.columns:
        # Cálculo de la cardinalidad y porcentaje de cardinalidad
        cardinalidad = dataframe[columna].nunique()
        porcentaje_cardinalidad = (cardinalidad / len(dataframe)) * 100

        # Sugerencia del tipo de variable
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                if dataframe[columna].dtype != 'object':
                    tipo_sugerido = "Numérica Continua"
                else:
                    tipo_sugerido = "Object"
            else:
                if dataframe[columna].dtype != 'object':
                    tipo_sugerido = "Numérica Discreta"
                else:
                    tipo_sugerido = "Object"
        # Agregar la sugerencia de tipo de variable a la lista
        sugerencias_tipos.append([columna, tipo_sugerido])

    # Construcción del DataFrame de sugerencias
    df_sugerencias = pd.DataFrame(sugerencias_tipos, columns=["nombre_variable", "tipo_sugerido"])

    # Retorno del DataFrame de sugerencias
    return df_sugerencias

#####################################################################################################################

### Funcion: get_features_num_regression 

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    
    """"
Esta función toma los siguientes argumentos:
Df: El DataFrame de pandas sobre el que realizará la función.
target_col: Es el nombre de la columna objetivo, el Target.
umbral_corr: Un umbral de correlación, entre 0 y 1.
Pvalue: Un valor que por defecto está desactivado.

La función irá comprobando las relaciones entre las columnas numéricas del DataFrame y la columna target, que también es numérica. 
Devolverá una lista con las columnas cuya correlación con el target sea superior a lo indicado en la variable umbral_corr. 
Además, si la columna pvalue está activada, el test de hipótesis entre las columnas y el target debe ser igual o superior a lo indicado en dicha variable.
"""

    
    # Comprobación de que df es un DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame de pandas.")
        return None
    
    # Comprobación de que target_col es una columna en el DataFrame
    if target_col not in df.columns:
        print("Error: 'target_col' no es una columna válida en el DataFrame.")
        return None
    
    # Comprobación de que target_col es numérica
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' no es una variable numérica en el DataFrame.")
        return None
    
    # Comprobación de que umbral_corr está entre 0 y 1
    if not 0 <= umbral_corr <= 1:
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None
    
    # Comprobación de que pvalue es None o un número entre 0 y 1
    if pvalue is not None and not 0 <= pvalue <= 1:
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None
    
    # Calcula la correlación de Pearson entre target_col y cada otra columna numérica
    corr_values = df.select_dtypes(include=np.number).apply(lambda x: pearsonr(df[target_col], x)[0])
    
    # Filtra las columnas con correlación mayor al umbral_corr
    relevant_features = corr_values[abs(corr_values) > umbral_corr].index.tolist()
    
    # Si pvalue no es None, filtra también por el valor de p-value
    if pvalue is not None:
        significant_features = []
        for feature in relevant_features:
            # Calcula el p-value para la correlación entre target_col y la feature actual
            _, p_val = pearsonr(df[target_col], df[feature])
            # Comprueba si el p-value es menor que 1-pvalue (significación mayor o igual a 1-pvalue)
            if p_val <= (1 - pvalue):
                significant_features.append(feature)
        return significant_features
    
    return relevant_features

#####################################################################################################################

### Funcion: plot_features_num_regression 

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
    for i in range(0, len(filtered_columns), 4):
        sns.pairplot(df[filtered_columns[i:i+4] + [target_col]], kind='reg', diag_kind='kde',plot_kws={'scatter_kws': {'s': 5}})
        plt.show()
    
    return filtered_columns
# Ejemplo de uso
#plot_features_num_regression(df_inmo, target_col="median_house_value", columns=[], umbral_corr=0.5, pvalue=0.05)

#####################################################################################################################

# Funcion: get_features_cat_regression


def get_features_cat_regression(dataframe, target_col, pvalue=0.05):
    '''
Selección de columnas categóricas del dataframe (features cat) cuyo test de relación con la columna target
supere en confianza estadística el test de relación que sea necesario, en este caso, chi2 y ANOVA.

Argumentos:
dataframe (DataFrame): DataFrame que contiene los datos.
target_col (str): Nombre de la columna que es el objetivo de la regresión (target).
pvalue (float): Nivel de significancia para el test estadístico. Por defecto 0.05.

Retorna:
list: Lista de columnas categóricas significantes.
'''
    # Comprobación de la existencia de la columna target_col
    if target_col not in dataframe.columns:
        print(f"Error: La columna {target_col} no existe en el DataFrame.")
        return None
    
    # Verificación de si target_col es una columna numérica continua del dataframe
    if target_col not in dataframe.columns or not np.issubdtype(dataframe[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es una columna numérica continua válida.")
        return None
    
    # Comprobación de que pvalue es un valor válido
    if not isinstance(pvalue, (int, float)) or pvalue <= 0 or pvalue >= 1:
        print(f"Error: {pvalue} debe ser un número entre 0 y 1.")
        return None
    
    # Obteneción de columnas categóricas
    cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Si no hay columnas categóricas, retornar None
    if not cat_cols:
        print("No hay columnas categóricas en el dataframe.")
        return None
    
    # Usar los tests para comprobar la relación entre las variables categóricas y target_col
    significant_features = []
    for col in cat_cols:
        # Para determinar qué test de relación utilizar, lo hacemos verificancdo el nº de niveles únicos (categorías) en col
        contingency_table = pd.crosstab(dataframe[col], dataframe[target_col])
        
        if contingency_table.shape[0] == 2:  # Test chi-cuadrado: si hay 2 niveles únicos, o sea, tablas de contigencia 2x2
            _, p_val, _, _ = chi2_contingency(contingency_table) #esto devuelve el estadístico de chi2 y el valor p
            
        else:  # Test ANOVA; se usa para más de 2 grupos
            _, p_val = f_oneway(*[dataframe[target_col][dataframe[col] == val] for val in dataframe[col].unique()]) # devuelve el valor F y el valor p
        
        # Comprobar si el p-valor es menor que el pvalue especificado. Si el valor p < pvalue, se rechaza la hipotesis nula y se considera que es una col cat significativa
        if p_val < pvalue:
            significant_features.append(col)
    
    return significant_features

#####################################################################################################################

# Funcion: plot_features_cat_regression 

def plot_features_cat_regression(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Función que genera los gráficos pairplot de las variables (columnas) de un DataFrame dada una variable target numérica.

    Argumentos:
    df (DataFrame): DataFrame que contiene las variables para las que queremos generar los gráficos pairplot.
    target_col (string): Nombre de la variable del DataFrame considerada como target.
    lista_columnas (lista) = Nombres de las columnas del DataFrame para las que queremos generar los gráficos pairplot
    umbral_corr (float) = valor mínimo de correlación para seleccionar las variables.
    umbral_pvalue (float) = valor máximo de pvalue para seleccionar las variables.
    limite_pairplot (int) = valor máximo de variables a generar en los gráficos pairplot.

    Retorna:
    Lista: devuelve una lista con los nombres de las columnas numéricas que cumplen las condiciones.
    """
    
    # Verificar los valores de entrada
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("El primer argumento debe ser un dataframe.")
    
    if target_col not in dataframe.columns:
        raise ValueError(f"La columna '{target_col}' no existe en el dataframe.")
    
    if not isinstance(columns, list):
        raise ValueError("El argumento 'columns' debe ser una lista.")
    
    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Al menos una de las columnas especificadas en 'columns' no existe en el dataframe.")
    
    if not isinstance(pvalue, (float, int)):
        raise ValueError("El argumento 'pvalue' debe ser un número.")
    
    if not isinstance(with_individual_plot, bool):
        raise ValueError("El argumento 'with_individual_plot' debe ser un valor booleano.")
    
    # Si la lista 'columns' está vacía, asignar las variables numéricas del dataframe
    if not columns:
        columns = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    # Almacenar las columnas que cumplen las condiciones
    significant_columns = []
    
    for col in columns:
        # Realizar el test de chi-cuadrado entre la variable categórica y la target
        contingency_table = pd.crosstab(dataframe[col], dataframe[target_col])
        _, p_val, _, _ = chi2_contingency(contingency_table)
        
        # Comprobar si el p-valor es significativo
        if p_val <= pvalue:
            significant_columns.append(col)
            
    # Si se especifica, plotear el histograma agrupado
    num_graficos = len(significant_columns) // 2
    
    if len(significant_columns) % 2 != 0:
        num_graficos = num_graficos + 1
    
    if with_individual_plot:
        fig, axs = plt.subplots(num_graficos, 2, figsize=(20, 20))
        axs= axs.flatten()
      
    # Recorrer la lista de nombres de columnas y crear textos en cada subgráfico
        for i in range(len(significant_columns)):
            sns.histplot(data= dataframe,x = target_col , hue = significant_columns[i], ax= axs[i], kde=True)

        if len(significant_columns) % 2 != 0: 
            axs[-1].axis("Off")
                
    return significant_columns

#Ejemplo: plot_features_cat_regression(dataframe=df_vinos, target_col="quality", columns= [], pvalue=0.05, with_individual_plot=False)

#####################################################################################################################

# TEAM CHALLENGE 17 #

#####################################################################################################################

### Funcion: eval_model (Alfonso)

def eval_model(target, predictions, problem_type, metrics):
    """
    Evalúa un modelo de regresión o clasificación en base a un conjunto de métricas especificadas.

    Parámetros:
    target : array-like
        Valores verdaderos del target.
    predictions : array-like
        Valores predichos por el modelo.
    problem_type : str
        Tipo de problema ('regression' o 'classification').
    metrics : list of str
        Lista de métricas a calcular. 
        - Para regresión: ['RMSE', 'MAE', 'MAPE', 'GRAPH']
        - Para clasificación: ['ACCURACY', 'PRECISION', 'RECALL', 'CLASS_REPORT', 'MATRIX', 'MATRIX_RECALL', 'MATRIX_PRED', 'PRECISION_X', 'RECALL_X']
          donde 'X' es una etiqueta de alguna de las clases del target.

    Retorna:
    tuple
        Tupla con los resultados de las métricas calculadas en el orden especificado en la lista de métricas. 
        Las métricas que no devuelven valores numéricos retornan None en su lugar.

    Excepciones:
    ValueError
        Se lanza si ocurre un error en el cálculo de una métrica, si se especifica una métrica no soportada,
        si el tipo de problema es inválido, o si se especifica una clase que no está presente en el target.

    Ejemplos de uso:
    >>> eval_model(y_true, y_pred, 'regression', ['RMSE', 'MAE', 'GRAPH'])
    >>> eval_model(y_true, y_pred, 'classification', ['ACCURACY', 'PRECISION', 'RECALL', 'MATRIX', 'PRECISION_1', 'RECALL_2'])
    """
    results = {}
    target = np.array(target)
    predictions = np.array(predictions)

    if problem_type == "regression":
        for metric in metrics:
            if metric == "RMSE":
                rmse = np.sqrt(mean_squared_error(target, predictions))
                print(f"RMSE: {rmse:.3f}")
                results['RMSE'] = round(rmse, 3)
            elif metric == "MAE":
                mae = mean_absolute_error(target, predictions)
                print(f"MAE: {mae:.3f}")
                results['MAE'] = round(mae, 3)
            elif metric == "MAPE":
                if np.any(target == 0):
                    print("MAPE no se puede calcular porque hay valores reales que son cero.")
                    results['MAPE'] = None
                else:
                    mape = mean_absolute_percentage_error(target, predictions)
                    print(f"MAPE: {mape:.3f}")
                    results['MAPE'] = round(mape, 3)
            elif metric == "GRAPH":
                plt.scatter(target, predictions)
                plt.xlabel("Valores Reales")
                plt.ylabel("Predicciones")
                plt.title("Valores Reales vs Predicciones")
                plt.show()
                results['GRAPH'] = None
            else:
                print(f"Métrica de regresión no soportada: {metric}")
                results[metric] = None

    elif problem_type == "classification":
        unique_classes = np.unique(target)
        print(f"Clases únicas en el objetivo: {unique_classes}")

        for metric in metrics:
            if metric == "ACCURACY":
                accuracy = accuracy_score(target, predictions)
                print(f"Accuracy: {accuracy:.3f}")
                results['ACC'] = round(accuracy, 3)
            elif metric == "PRECISION":
                precision = precision_score(target, predictions, average='weighted', zero_division=0)
                print(f"Precision: {precision:.3f}")
                results['PREC'] = round(precision, 3)
            elif metric == "RECALL":
                recall = recall_score(target, predictions, average='weighted', zero_division=0)
                print(f"Recall: {recall:.3f}")
                results['REC'] = round(recall, 3)
            elif metric == "CLASS_REPORT":
                report = classification_report(target, predictions)
                print("Informe de Clasificación:\n", report)
                results['REPORT'] = None
            elif metric == "MATRIX":
                matrix = confusion_matrix(target, predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
                disp.plot()
                plt.show()
                results['MATRIX'] = None
            elif metric.startswith("PRECISION_") or metric.startswith("RECALL_"):
                class_label = metric.split("_")[1]
                if class_label.isdigit() and int(class_label) in unique_classes:
                    if metric.startswith("PRECISION_"):
                        precision_class = precision_score(target, predictions, labels=[int(class_label)], average='weighted', zero_division=0)
                        print(f"Precision para la clase {class_label}: {precision_class:.3f}")
                        results[f'PREC_{class_label}'] = round(precision_class, 3)
                    elif metric.startswith("RECALL_"):
                        recall_class = recall_score(target, predictions, labels=[int(class_label)], average='weighted', zero_division=0)
                        print(f"Recall para la clase {class_label}: {recall_class:.3f}")
                        results[f'REC_{class_label}'] = round(recall_class, 3)
                else:
                    print(f"La etiqueta de clase {class_label} no se encuentra en las etiquetas objetivo. Clases disponibles: {unique_classes}")
                    results[f'{metric}_{class_label}'] = None
            else:
                print(f"Métrica de clasificación no soportada: {metric}")
                results[metric] = None

    else:
        raise ValueError("Por favor, usa 'regression' o 'classification' para el tipo de problema.")

    return results



#####################################################################################################################

### Funcion: get_features_num_classification (Pepe)

def get_features_num_classification(df, target_col, p_value= 0.05): 
    """Esta función toma como argumentos un dataframe de pandas, una columna que será la columna objetivo (el target) 
    y un valor de p-value que por defecto es 0.05.

    La función verifica si el primer argumento es un dataframe válido y si la columna objetivo está presente en dicho dataframe. 
    Luego, selecciona las columnas numéricas del dataframe que no son la columna objetivo y verifica si la columna objetivo 
    cumple con los tipos de datos válidos para el análisis (categórico, entero, booleano u objeto).

    Posteriormente, realiza un test ANOVA para cada columna numérica seleccionada, evaluando si existe una relación 
    estadísticamente significativa con la columna objetivo basada en el p-value proporcionado. Devuelve una lista de 
    columnas numéricas que cumplen con el criterio de relación ANOVA especificado por el p-value.
    """
    columnas_num= [] 
    columnas_validas= []
    #limite = 1- p_value #No stoy seguro si hacerlo asi o directamente p_value 
    if isinstance(df, pd.DataFrame):
        print(f"El primer termino es un DataFrame válido") 
    else:
        print(f"El primer termino introducido no es un dataframe, repase la llamada a la función")
        return 

    if target_col not in df.columns:
        print(f"{target_col} no es una columna del dataframe df")
        return
    
    tipos_validos = ['category', 'int64', 'bool', 'object']
    tipo_columna = df[target_col].dtype
    if tipo_columna not in tipos_validos:
        print(f"La columna '{target_col}', especificada en la llamada como target, no es categórica ni discreta, revisa la llamada a la funcion.")
        return 
    
    if pd.api.types.is_integer_dtype(df[target_col]) and df[target_col].nunique() >= 15: # Este "15" es el maximo a partir del cual avisa de la alta cardinalidad.
        print(f"¡OJO!.. La columna '{target_col}' tiene alta cardinalidad (> 15 categorías).")
    
    columnas_pre_num = df.select_dtypes(include=['number']).columns.tolist() 
    columnas_num= [] 
    
    for columna_numericas in columnas_pre_num:
        if columna_numericas != target_col: #Con esto nos aseguramos de no incluir el target.
            columnas_num.append(columna_numericas)
        elif columna_numericas == target_col:
            print("La columna objetivo no ha sido incluida en la lista.")
    
    if any(df[column].isnull().any() for column in columnas_num):
        print("ALGUNA DE SUS COLUMNAS NUMÉRICAS TIENE DATOS FALTANTES O ERRÓNEOS, LIMPIE SU DATAFRAME ANTES DE CONTINUAR")
        return
    
    for columna in columnas_num:
        grupos = []
        for categoria in df[target_col].unique():
            grupos.append(df[columna][df[target_col] == categoria])
    
        anova_result = f_oneway(*grupos)
        valor_de_p = anova_result.pvalue
        if valor_de_p <= 1-p_value: 
            columnas_validas.append(columna)


    cantidad_elementos = len(columnas_validas)
    if cantidad_elementos == 0:
        print("NO HAY COLUMNAS QUE CUMPLAN LOS REQUISITOS")
        return
    if cantidad_elementos == 1:
        print("Solo una columna cumple los requisitos:")
        print(f"El valor de pvalue es {anova_result.pvalue}")
        return columnas_validas
    if cantidad_elementos > 1:
        print("Las columnas que cumplen requisitos son:")
        return columnas_validas


#####################################################################################################################

### Funcion: plot_features_num_classification (María)
def plot_features_num_classification(df, target_col="", columns=[], pvalue=0.05):
    """
    Descripción: Realiza un análisis de clasificación entre una columna objetivo y las columnas numéricas de un DataFrame,
    filtrando aquellas que tienen un valor p bajo según el test de ANOVA.

    Argumentos:
    df (DataFrame): El DataFrame que contiene los datos.
    target_col (str): El nombre de la columna objetivo que se usará en el análisis de clasificación.
    columns (list): Una lista de nombres de columnas a considerar. Si está vacía, se considerarán todas las columnas numéricas del DataFrame.
    pvalue (float): El valor p máximo para considerar una columna como estadísticamente significativa.

    Retorna:
    list: Columnas que cumplen con los requisitos de significancia estadística.
    """
    # Verificación de la columna objetivo y obtención de columnas válidas
    if not target_col:
        raise ValueError("El argumento 'target_col' no puede estar vacío.")
    
    valid_columns = get_features_num_classification(df, target_col, p_value=pvalue)
    if valid_columns is None:
        return []

    # Si columns no está vacío, filtrar valid_columns para mantener solo las especificadas en columns
    if columns:
        valid_columns = [col for col in valid_columns if col in columns]

    # Obtener valores únicos de target_col
    unique_target_values = df[target_col].unique()
    
    # Dividir las columnas en grupos de máximo cuatro adicionales, incluyendo siempre target_col para un total de cinco
    for i in range(0, len(valid_columns), 5):
        selected_columns = valid_columns[i:i+5]
        columns_to_plot = [target_col] + selected_columns  # Siempre incluir target_col
        
        for j in range(0, len(unique_target_values), 5):
            current_target_values = unique_target_values[j:j+5]
            df_filtered = df[df[target_col].isin(current_target_values)]
            
            g = sns.pairplot(df_filtered[columns_to_plot], 
                             hue=target_col, 
                             kind='reg', 
                             diag_kind='kde',
                             plot_kws={'scatter_kws': {'s': 10}},  # Tamaño de los puntos ajustado
                             height=2.5)  # Tamaño de la figura ajustado
            
            # Ajustar los títulos y las etiquetas de los ejes
            g.fig.suptitle(f'{target_col} y columnas: {", ".join(selected_columns)}', y=1.02)
            for ax in g.axes.flatten():
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            
            plt.show()
    
    return valid_columns

# Ejemplo de uso
# plot_features_num_classification(df, target_col="target", columns=[], pvalue=0.05)

#####################################################################################################################

### Funcion: get_features_cat_classification (Brenda)

def get_features_cat_classification(df, target_col, normalize=False, mi_threshold=0):
    """
    Selecciona las columnas categóricas en un DataFrame cuyo valor de información mutua con respecto a la columna objetivo
    sea mayor o igual a un umbral especificado.

    Argumentos:
        df (pd.DataFrame): El DataFrame de entrada.
        target_col (str): El nombre de la columna objetivo.
        normalize (bool): Si True, normaliza los valores de información mutua. Por defecto es False.
        mi_threshold (float): El umbral de información mutua para seleccionar características. Por defecto es 0.0.

    Retorna:
        list: Lista de las columnas categóricas que cumplen con el umbral de información mutua.
    """



    # Comprobaciones de entrada
    if target_col not in df.columns:
        print("Error: target_col no está en el dataframe.")
        return None
    
    if not isinstance(df[target_col].dtype, pd.CategoricalDtype) and not pd.api.types.is_object_dtype(df[target_col]):
        print("Error: target_col debe ser una variable categórica.")
        return None
    
    if normalize and not (0 <= mi_threshold <= 1):
        print("Error: mi_threshold debe ser un float entre 0 y 1 cuando normalize es True.")
        return None
    
    # Selección de columnas categóricas
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in cat_columns:
        cat_columns.remove(target_col)
    
    if len(cat_columns) == 0:
        print("Error: No hay columnas categóricas en el dataframe.")
        return None
    
    # OneHotEncoding de las columnas categóricas
    encoder = OneHotEncoder(drop='first')
    X_encoded = encoder.fit_transform(df[cat_columns])
    
    # Cálculo de la mutual information
    mi = mutual_info_classif(X_encoded, df[target_col])
    
    if normalize:
        mi_sum = np.sum(mi)
        if mi_sum == 0:
            print("Error: La suma de la mutual information es cero, no se puede normalizar.")
            return None
        mi_normalized = mi / mi_sum
        selected_columns = [col for col, val in zip(cat_columns, mi_normalized) if val >= mi_threshold]
    else:
        selected_columns = [col for col, val in zip(cat_columns, mi) if val >= mi_threshold]
    
    return selected_columns


#####################################################################################################################

### Funcion: plot_features_cat_classification (Fernando)


def plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False):
    """
    Grafica la distribución de las clases objetivo para características categóricas en un dataframe,
    filtradas por umbral de información mutua con respecto a la columna objetivo.

    Argumentos:
    df (DataFrame): DataFrame de entrada que contiene características categóricas y la columna objetivo.
    target_col (str): Nombre de la columna objetivo. Por defecto es "".
    columns (list of str): Lista de nombres de columnas a considerar. Si está vacía,
                           se considerarán todas las columnas categóricas en df. Por defecto es [].
    mi_threshold (float): Valor de umbral para la información mutua. Las características con MI
                          mayor que este umbral se considerarán. Por defecto es 0.0.
    normalize (bool): Si se debe normalizar los conteos en el gráfico. Por defecto es False.

    Retorna:
    None
    """
    
    # Verifica si se proporcionó target_col y si está en el dataframe
    if target_col != "" and target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no se encontró en las columnas del dataframe.")

    # Si columns está vacía, selecciona todas las columnas categóricas de df
    if not columns:
        columns = [col for col in df.columns if df[col].dtype == 'object']

    # Valida que las columnas existan en el dataframe
    invalid_cols = [col for col in columns if col not in df.columns]
    if invalid_cols:
        raise ValueError(f"Las columnas {invalid_cols} no se encontraron en las columnas del dataframe.")

    # Selecciona las columnas categóricas basadas en el umbral de información mutua si está especificado
    selected_columns = []
    for col in columns:
        if target_col != "":
            mi = mutual_info_score(df[col], df[target_col])
            if mi > mi_threshold:
                selected_columns.append(col)
        else:
            selected_columns.append(col)

    # Grafica las distribuciones para las columnas categóricas seleccionadas
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        if normalize:
            sns.countplot(x=col, hue=target_col, data=df, palette='Set2', edgecolor='black')
            plt.ylabel('Conteo Normalizado')
        else:
            sns.countplot(x=col, hue=target_col, data=df, palette='Set2', edgecolor='black')
            plt.ylabel('Conteo')
        plt.title(f'Distribución de {col} con respecto a {target_col}')
        plt.xticks(rotation=45)
        plt.legend(title=target_col)
        plt.tight_layout()
        plt.show()



#####################################################################################################################

### EXTRA: Funcion: super_selector

def super_selector(dataset, target_col="", selectores=None, hard_voting=[]):
    """
    Función Super Selector para seleccionar características de un dataset basándose en varios métodos de selección de características.

    Parámetros:
    dataset (pd.DataFrame): El dataframe de entrada que contiene las características y la columna objetivo.
    target_col (str): El nombre de la columna objetivo. Si está vacío, no se realizará una selección basada en el objetivo.
    selectores (dict): Un diccionario que especifica los métodos de selección de características y sus parámetros.
        Claves posibles y sus valores esperados:
            - "KBest": int, número de características a seleccionar usando SelectKBest con ANOVA o f_regression.
            - "FromModel": lista, que contiene un estimador y un valor de umbral o un entero para características máximas.
            - "RFE": tupla, que contiene un estimador, número de características a seleccionar y valor de step.
            - "SFS": tupla, que contiene un estimador y número de características a seleccionar usando Sequential Feature Selector.
    hard_voting (list): Una lista inicial de características para incluir en el proceso de hard voting.

    Retorno:
    dict: Un diccionario con claves correspondientes a los selectores proporcionados y sus listas de características seleccionadas,
          y una clave adicional "hard_voting" con las características más votadas en todos los métodos.
          
    Ejemplo:
    >>> selectores = {
    >>>     "KBest": 5,
    >>>     "FromModel": [RandomForestClassifier(), 5],
    >>>     "RFE": [LogisticRegression(max_iter=5000), 5, 1],
    >>>     "SFS": [RandomForestClassifier(), 5]
    >>> }
    >>> 
    >>> result = super_selector(dataset, target_col="Survived", selectores=selectores, hard_voting=[])
    >>> print(result)
    """
    if selectores is None:
        selectores = {}

    # Validamos que target_col sea una columna válida del dataframe
    if target_col and target_col not in dataset.columns:
        raise ValueError(f"{target_col} is not a valid column in the dataset")

    # Lista de todas las features, excluyendo target_col
    feature_cols = [col for col in dataset.columns if col != target_col]
    X = dataset[feature_cols]
    y = dataset[target_col] if target_col else None

    # Identificar variables categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Crear preprocesador con OneHotEncoder para categóricas y StandardScaler para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )

    # Ajustar el preprocesador a los datos
    preprocessor.fit(X)

    # Determinar si la variable objetivo es categórica o numérica
    target_is_categorical = y.dtype == 'object' or y.dtype.name == 'category'

    selected_features = {}

    if not selectores:
        # Si selectores es un diccionario vacío o None
        features = []
        for col in feature_cols:
            unique_vals = dataset[col].nunique()
            if unique_vals == 1 or unique_vals == len(dataset):
                continue
            features.append(col)
        selected_features['all_features'] = features
    else:
        def get_transformed_feature_names(column_transformer):
            """Get feature names after transformation."""
            output_features = []
            for name, transformer, columns in column_transformer.transformers_:
                if transformer == 'drop' or isinstance(transformer, str):
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    # Asegurarse de que el transformador esté ajustado antes de llamar a get_feature_names_out
                    if not hasattr(transformer, 'categories_'):
                        transformer.fit(X[columns])
                    names = transformer.get_feature_names_out(columns)
                else:
                    names = columns
                output_features.extend(names)
            return output_features

        if "KBest" in selectores:
            k = selectores["KBest"]
            score_func = f_classif if target_is_categorical else f_regression
            kbest_selector = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(score_func, k=k))
            ])
            kbest_selector.fit(X, y)
            kbest_feature_indices = kbest_selector.named_steps['selector'].get_support(indices=True)
            transformed_feature_names = get_transformed_feature_names(preprocessor)
            kbest_features = [transformed_feature_names[i] for i in kbest_feature_indices]
            selected_features["KBest"] = kbest_features

        if "FromModel" in selectores:
            model, threshold = selectores["FromModel"]
            if isinstance(threshold, int):
                sfm_selector = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('selector', SelectFromModel(model, max_features=threshold, threshold=-np.inf))
                ])
            else:
                sfm_selector = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('selector', SelectFromModel(model, threshold=threshold))
                ])
            sfm_selector.fit(X, y)
            sfm_feature_indices = sfm_selector.named_steps['selector'].get_support(indices=True)
            transformed_feature_names = get_transformed_feature_names(preprocessor)
            sfm_features = [transformed_feature_names[i] for i in sfm_feature_indices]
            selected_features["FromModel"] = sfm_features

        if "RFE" in selectores:
            model, n_features_to_select, step = selectores["RFE"]
            rfe_selector = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', RFE(model, n_features_to_select=n_features_to_select, step=step))
            ])
            rfe_selector.fit(X, y)
            rfe_feature_indices = rfe_selector.named_steps['selector'].get_support(indices=True)
            transformed_feature_names = get_transformed_feature_names(preprocessor)
            rfe_features = [transformed_feature_names[i] for i in rfe_feature_indices]
            selected_features["RFE"] = rfe_features

        if "SFS" in selectores:
            model, k_features = selectores["SFS"]
            sfs_selector = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', SequentialFeatureSelector(model, n_features_to_select=k_features, direction='forward'))
            ])
            sfs_selector.fit(X, y)
            sfs_feature_indices = sfs_selector.named_steps['selector'].get_support(indices=True)
            transformed_feature_names = get_transformed_feature_names(preprocessor)
            sfs_features = [transformed_feature_names[i] for i in sfs_feature_indices]
            selected_features["SFS"] = sfs_features

    # Hard voting
    def hard_voting_feature_selection(selected_features):
        """Realiza una selección de características basada en hard voting."""
        # Diccionario para contar los votos de cada característica
        feature_votes = {}

        # Recorrer cada lista de características seleccionadas y asignar votos
        for feature_list in selected_features.values():
            for feature in feature_list:
                if feature in feature_votes:
                    feature_votes[feature] += 1
                else:
                    feature_votes[feature] = 1

        # Seleccionar las características con más votos
        max_votes = max(feature_votes.values())
        most_voted_features = [feature for feature, votes in feature_votes.items() if votes == max_votes]

        return most_voted_features

    most_voted_features = hard_voting_feature_selection(selected_features)
    selected_features["hard_voting"] = most_voted_features

    return selected_features


# Ejemplo de uso con variable categórica
selectores_categorico = { 
    "KBest": 5,
    "FromModel": [RandomForestClassifier(), 5],
    "RFE": [LogisticRegression(max_iter=5000), 5, 1],
    "SFS": [RandomForestClassifier(), 5] 
}

# Ejemplo de uso con variable numérica
selectores_numerico = { 
    "KBest": 5,
    "FromModel": [RandomForestRegressor(), 5],
    "RFE": [LinearRegression(), 5, 1],
    "SFS": [RandomForestRegressor(), 5] 
}

# Llamada a la función con un dataset y la variable objetivo categórica
# super_selector(df, target_col=" ", selectores=selectores_categorico, hard_voting=[])

# Llamada a la función con un dataset y la variable objetivo numérica
# super_selector(df, target_col="some_numeric_target", selectores=selectores_numerico, hard_voting=[])




#####################################################################################################################