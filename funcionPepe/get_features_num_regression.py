import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
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








