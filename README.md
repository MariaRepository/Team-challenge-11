<div style="display: flex; justify-content: center;">
    <img src="./imagenes/ban.png" style="border-radius: 18px; width: 95%;">
</div> 
  
<br>

### ¿QUÉ ENCONTRARÁS EN ESTE REPOSITORIO?

En este repositorio encontrarás una solución al "team chanllenge" planteado en el archivo `Team_Challenge_ToolBox_I.ipynb`. En otras palabras, incluye una serie de funciones Python para la creación de modelos de machine learning.  
Las funciones mencionadas están disponibles en el archivo **"toolbox_ML.py"**. 🧰🛠️

### LAS FUNCIONES SON LAS SIGUIENTES:

**describe_df** 
> Esta función toma un dataframe como argumento y devuelve otro dataframe que contiene información sobre cada columna. Esto incluye el tipo de datos, el porcentaje de valores nulos, la cantidad de valores únicos y la cardinalidad de cada columna.


**tipifica_variables**  
>"Esta función, que recibe 3 argumentos (dataframe, umbral_categoria y umbral_continua ) analiza un dataframe y devuelve los  tipos de variables de cada columna. Según la cardinalidad y umbrales dados (`umbral_categoria` y `umbral_continua`), los tipos son: Binaria, Categórica, Numérica Continua o Numérica Discreta."

**get_features_num_regression**  
>Esta función recibe como argumentos un dataframe, el nombre de una columna ('target_col') la columna target o objetivo, un umbral de correlación ('umbral_corr') y un valor de p-valor opcional ('pvalue'). Devuelve una lista de columnas numéricas del dataframe cuya correlación con 'target_col' supere 'umbral_corr'. Si se proporciona 'pvalue', solo se incluyen aquellas columnas que superen el test de hipótesis con una significación mayor o igual a 1-pvalue.  

**plot_features_num_regression**  
>Esta función recibe como argumentos un dataframe, el nombre de la columna objetivo (el target) "target_col" (por defecto ""), una lista de columnas ("columns", por defecto vacía), un umbral de correlación ("umbral_corr", por defecto 0) y un valor de p-valor ("pvalue", por defecto "None").
>
>Si la lista no está vacía, la función muestra un pairplot del dataframe considerando "target_col" y las columnas de "columns" cuya correlación con "target_col" supere "umbral_corr" (y el test de correlación, si se proporciona "pvalue"). Devuelve las columnas que cumplen estas condiciones.


**get_features_cat_regression**  
>Esta función recibe como argumentos un dataframe, el nombre de la columna objetivo (el target) "target_col"(debe ser una variable numérica continua o discreta con alta cardinalidad), y un valor de p-valor ('pvalue', por defecto 0.05).
>
>Devuelve una lista de columnas categóricas cuyo test de relación con 'target_col' supera el nivel de confianza estadística requerido, utilizando el test adecuado. Se realizan comprobaciones para garantizar valores adecuados, incluyendo que 'target_col' sea una variable numérica continua del dataframe.


**plot_features_cat_regression** 
>Esta función recibe un dataframe, un argumento "target_col" (por defecto ""), una lista de columnas ("columns", por defecto vacía), un argumento "pvalue" (por defecto 0.05), y un argumento "with_individual_plot" (por defecto False).
>
>Muestra histogramas agrupados de "target_col" para cada valor de las variables categóricas en "columns", si su test de relación es significativo. Devuelve las columnas que cumplen estas condiciones.
>
>Se realizan comprobaciones para garantizar valores adecuados de entrada.    
<br>  

👁️ TANTO EN EL ARCHIVO DONDE SE PLANTEA EL RETO, **'Team_Challenge_ToolBox_I.ipynb'** como en el archivo python **'toolbox_ML.py'** LAS FUNCIONES ESTÁN EXPLICADAS, DE FORMA MAS EXTENSA.

<img src="imagenes/caJin.jpg" alt="Descripción de la imagen" style="border-radius: 10px; width: 65%;">  

<br>    

Si, las funciones funcionan. Han sido probadas con varios dataset. 
En el archivo **"prueba.ipynb"** se carga el dataframe [AQUI PONEMOS LO QUE TOCA] y se importa el archivo 'toolbox_ML.py'  para ir llamando a las distintas funciones. 



  
**TRABAJO REALIZADO POR:** 

- [**Alfonso Nieto García**](https://github.com/ANG112)
- [**Brenda Rodríguez Farfán**](https://github.com/BrendzRdgz)
- [**Fernando Manzano**](https://github.com/FernandoManzanoC)
- [**María Fernández**](https://github.com/MariaRepository)
- [**Pepe Reina Campo**](https://github.com/PepeReinaCampo )
