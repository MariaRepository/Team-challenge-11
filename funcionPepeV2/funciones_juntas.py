import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


### Funcion: describe_df (Alfonso)

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

### Funcion: tipifica_variables (Alfonso)

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
                tipo_sugerido = "Numérica Continua"
            else:
                tipo_sugerido = "Numérica Discreta"

        # Agregar la sugerencia de tipo de variable a la lista
        sugerencias_tipos.append([columna, tipo_sugerido])

    # Construcción del DataFrame de sugerencias
    df_sugerencias = pd.DataFrame(sugerencias_tipos, columns=["nombre_variable", "tipo_sugerido"])

    # Retorno del DataFrame de sugerencias
    return df_sugerencias

#####################################################################################################################

### Funcion: get_features_num_regression (Pepe)

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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

### Funcion: get_features_num_regression VERSION 2 (Pepe)

def get_features_num_regression_V2(df, target_col, umbral_corr, pvalue=None):
    
    """
    Esta función toma los siguientes argumentos:
    Df: El DataFrame de pandas sobre el que realizará la función.
    target_col: Es el nombre de la columna objetivo, el Target.
    umbral_corr: Un umbral de correlación, entre 0 y 1.
    Pvalue: Un valor que por defecto está desactivado.

    La función irá comprobando las relaciones entre las columnas numéricas del DataFrame y la columna target, que también es numérica. 
    Devolverá una lista con las columnas cuya correlación con el target sea superior a lo indicado en la variable umbral_corr. 
    Además, si la columna pvalue está activada, el test de hipótesis entre las columnas y el target debe ser igual o superior a lo indicado en dicha variable.
    """    
    
    #Primero comprobamos que el primer atributo, df, sea un DataFrame. 
    if not isinstance(df, pd.DataFrame):
        print(f"{df} NO ES UN DATAFRAME DE PANDAS, POR FAVOR REVISA LA LLAMADA A LA FUNCION") 
        return None
    #Ahora comprobamos que la segunda variable de la funcion sea una columna numérica de del dataframe.
    nombres_columnas = list(df.columns) #Creamos una lista con los nombres de las columnas.
    
    
    if target_col not in nombres_columnas:
        print(f"{target_col} no es una columna del DataFrame df, por favor revise la llamada a la función.")
        print("COMPRUEBE SUS DATOS")
        return 
    else:
        tipo_de_datos = df[target_col].dtype
        if tipo_de_datos in ['int64', 'float64']: #Vemos que es una columna numerica
            print()
        else:
            try:
                pd.to_numeric(df[target_col]) #Con esto vemos que la columna se puede convertir a numerica. 
                print(f"{target_col} HA SIDO CONVERTIDA A TIPO NUMÉRICO")
            except ValueError:
                print(f"{target_col} NO ES DE TIPO NUMERICO NI SE PUEDE CONVERTIR EN TIPO NUMERICO.")
                print("COMPRUEBE SUS DATOS")
                return None
    #Comprobamos que el valor introduciodo como umbral de correlacion sea valido.
    if not str(umbral_corr).replace('.', '', 1).isdigit():
        print(F"{umbral_corr} NO ES UN NUMERO, EL 3ª TERMINO DE ESTA FUNCION DEBE SER UN NUMERO ENTRE 0 Y 1")
        return ("COMPRUEBE SUS DATOS")
    else:
        if umbral_corr > 1:
            print("EL UMBRAL DE CORRELACION NO PUEDE SER MAYOR A 1")
            return ("COMPRUEBE SUS DATOS")
     
        if   umbral_corr < 0:
                print("EL UMBRAL DE CORRELACION NO PUEDE SER MENOR A 0")
                return ("COMPRUEBE SUS DATOS")
        else:
            print() 
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)):
            print("El valor pvalue debe ser un número.")
            return None
        elif pvalue <= 0 or pvalue >= 1:
            print("El valor pvalue debe estar entre 0 y 1.")
            return None 
        else: 
            #Ahora buscamos las columnas con un correlacion con el target superior al umbral.
            #Primero creamos una lista con todas las columnas numericas (menos el target) y el test de hipotesis mayor o igual que p-value
            columnas_numericas = []
            for columna in df.columns:
                 if pd.api.types.is_numeric_dtype(df[columna]) and columna != target_col:
                     columnas_numericas.append(columna)
            columnas_numericas_umbral = []
            for columna_umbral in columnas_numericas:
                correlacion = df[columna_umbral].corr(df[target_col])
                if abs(correlacion) > umbral_corr and stats.ttest_ind(df[columna_umbral], df[target_col]).pvalue <= pvalue:
                    columnas_numericas_umbral.append(columna_umbral)
            print("Las columnas numéricas cuya correlación con el target es superior a lo que necesitamos son:")
            return columnas_numericas_umbral 

    else: 
        #Ahora buscamos las columnas con un correlacion con el target superior al umbral.
    #Primero creamos una lista con todas las columnas numericas (menos el target)
   
        columnas_numericas = []
        for columna in df.columns:
            if pd.api.types.is_numeric_dtype(df[columna]) and columna != target_col:
                columnas_numericas.append(columna) 
    #Si la corralacion (en valor absoluto) de la columna con el target es superior al umbral de correlacion, se añadirá a una lista que será la que devulva la funcion.
        columnas_numericas_umbral = []
        for columna_umbral in columnas_numericas:
            correlacion = df[columna_umbral].corr(df[target_col])
            if abs(correlacion) > umbral_corr:
                columnas_numericas_umbral.append(columna_umbral)
        print("Las columnas numéricas cuya correlación con el target es superior a lo que necesitamos son:")     
        return columnas_numericas_umbral 

#####################################################################################################################

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
    for i in range(0, len(filtered_columns), 4):
        sns.pairplot(df[filtered_columns[i:i+4] + [target_col]], kind='reg', diag_kind='kde')
        plt.show()
    
    return filtered_columns
# Ejemplo de uso
#plot_features_num_regression(df_inmo, target_col="median_house_value", columns=[], umbral_corr=0.5, pvalue=0.05)

#####################################################################################################################

# Funcion: get_features_cat_regression (Brenda)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway


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

#####################################################################################################################

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
        sns.histplot(data=dataframe, x=feature, hue=target_col, multiple="stack", palette= "viridis")
        plt.title(f"Histograma agrupado para {feature} en relación con {target_col}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        plt.legend(title=target_col)
        plt.show()

        # Plot de histogramas individuales si se especifica
        if with_individual_plot:
            for value in dataframe[feature].unique():
                plt.figure(figsize=(6, 4))
                sns.histplot(data=dataframe[dataframe[feature] == value], x=target_col, palette= 'green')
                plt.title(f"Histograma de {target_col} para {feature}={value}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()
    
    return significant_features 


# Nuevo funcion plot num
def plot_features_num_regressionv2(df, target_col, lista_columnas="", umbral_corr=0, umbral_pvalue=None, limite_pairplot=5):
    
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
    
    #Comprobamos que los valores de entrada tienen el tipo correcto

    #Comprobamos para el argumento df
    error_df = False
    if not(isinstance(df, pd.DataFrame)):
        error_df = True

    #Comprobamos para el argumento target_col
    error_target_col = False
    if not(isinstance(target_col, str)):
        error_target_col = True

    if target_col not in df.columns.tolist():
        error_target_col = True
    
    elif df[target_col].dtypes != float:
        error_target_col = True

    #Comprobamos para el argumento lista_columnas
    error_lista_columnas = False
    if not(isinstance(lista_columnas, (list,str))):
        error_lista_columnas = True

    if lista_columnas == []:
        error_lista_columnas = True

    for columna in lista_columnas:
        
        if columna == target_col:
            error_lista_columnas = True

        if columna not in df.columns.tolist():
            error_lista_columnas = True

    #Comprobamos para el argumento umbral_corr
    error_umbral_corr = False
    if not(isinstance(umbral_corr, (int, float))):
        error_umbral_corr = True
        
    elif umbral_corr < 0 or umbral_corr > 1:
        error_umbral_corr = True

    #Comprobamos para el argumento umbral_pvalue
    error_umbral_pvalue = False
    if not(isinstance(umbral_pvalue, (int, float))):
        if umbral_pvalue != None:
            error_umbral_pvalue = True
    
    elif umbral_pvalue < 0 or umbral_pvalue > 1:
        error_umbral_pvalue = True

    #Comprobamos para el argumento limite_pairplot
    error_limite_pairplot = False
    if not(isinstance(limite_pairplot, int)):
        error_limite_pairplot = True

    elif limite_pairplot < 2:
        error_limite_pairplot = True

    #Comprobamos si ha habido algun un error y mostramos el mensaje de error en cada caso
    error = error_df or error_target_col or error_lista_columnas or error_umbral_corr or error_umbral_pvalue or error_limite_pairplot
    if error: 
        if error_df:
            print("Introduce un DataFrame")
        elif error_target_col:
            print("Introduce un string con el nombre de la variable numérica target del DataFrame") 
        elif error_lista_columnas:
            print("Introduce una lista con los nombres de las columnas (sin incluir el target) del DataFrame a analizar")   
        elif error_umbral_corr:
            print("Introduce un valor entero o decimal para el umbral de correlación y que esté comprendido entre 0.0 y 1.0")   
        elif error_umbral_pvalue:
            print("Introduce un valor entero o decimal para el umbral del pvalue y que esté comprendido entre 0.0 y 1.0")
        elif error_limite_pairplot:
            print("Introduce un valor entero con el límite de variables a representar en el pairplot y que sea mayor o igual a 2")
    
        return None
        
    else:

        if lista_columnas == "":
            #Creamos una lista con las columnas numericas del DataFrame
            lista_columnas_numericas = []

            for columna in df:
                
                try: 
                    pd.to_numeric(df[columna], errors='raise')
                    lista_columnas_numericas.append(columna) 
                
                except:
                    pass
            
            #Creamos el DataFrame de correlaciones con el target y nos quedamos con las variables que superan el umbral  
            df_corr = df[lista_columnas_numericas].corr()[[target_col]].abs()
            lista_columnas_pairplot = df_corr[df_corr[target_col] > umbral_corr].index.tolist()

        else: 
            
            lista_columnas.append(target_col)
            df_corr = df[lista_columnas].corr()[[target_col]].abs()
            lista_columnas_pairplot = df_corr[df_corr[target_col] > umbral_corr].index.tolist()

        #Comprobamos si supera el test de correlación
        if umbral_pvalue != None:
            
            for columna in lista_columnas_pairplot.copy():
                #Calculamos el pvalue y comprobamos su valor
                pvalue = pearsonr(df[target_col], df[columna]).pvalue
                
                #Si el pvalue es mayor que el umbral se elimina la columna
                if pvalue > umbral_pvalue:
                    lista_columnas_pairplot.remove(columna)

        #Eliminamos el target de la lista de columnas para calcular cuántos gráficos hay que generar
        lista_columnas_pairplot.remove(target_col)
        num_graficos = (len(lista_columnas_pairplot) // (limite_pairplot-1))

        if len(lista_columnas_pairplot) % (limite_pairplot-1) != 0:
            num_graficos = num_graficos + 1

        #Generamos los gráficos
        if num_graficos == 1:
            
            lista = lista_columnas_pairplot.copy()
            lista.append(target_col)
            
            sns.pairplot(df[lista], height=3, aspect=1.5)
            
        else:
            
            for i in range(0,num_graficos):
                #Creamos listas con las distintas columnas a relacionar con el target
                lista = lista_columnas_pairplot[(limite_pairplot-1)*i:(limite_pairplot-1)*i+(limite_pairplot-1)].copy()
                lista.append(target_col)
                
                sns.pairplot(df[lista], height=2, aspect=1.5)

        return lista_columnas_pairplot

#Ejemplo: plot_features_num_regression(df=df_practica, target_col="MEDV", lista_columnas="", umbral_corr=0.6, umbral_pvalue=0.05, limite_pairplot=5)


# Nueva funcion plot cat

def plot_features_cat_regressionv2(dataframe, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
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
