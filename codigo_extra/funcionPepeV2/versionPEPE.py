import pandas as pd



def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    #Primero comprobamos que el primer atributo, df, sea un DataFrame. 
    try:
        df.columns
    except AttributeError:
        print(f"{df} NO ES UN DATAFRAME DE PANDAS, POR FAVOR REVISA LA LLAMADA A LA FUNCION") 

    #Ahora comprobamos que la segunda variable de la funcion sea una columna numérica de del dataframe.
    nombres_columnas = df.columns.tolist() #Creamos una lista con los nombres de las columnas.
    if target_col not in nombres_columnas:
        print(f"{target_col} no es una columna del DataFrame {df}, por favor revise la llamada a la función.")
        return ("COMPRUEBE SUS DATOS")
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
                return ("COMPRUEBE SUS DATOS")
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

    #Ahora buscamos las columnas con un correlacion con el target superior al umbral.
    columnas_numericas = []
    for columna in df.columns:
        if pd.api.types.is_numeric_dtype(df[columna]) and columna != target_col:
            columnas_numericas.append(columna) 
    
    columnas_numericas_umbral = []
    for columna_umbral in columnas_numericas: