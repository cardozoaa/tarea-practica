import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

#metodo para cargar los datos
@st.cache_data

def cargar_datos(archivo):
    if archivo:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith('.xlsx'):
            df = pd.read_excel(archivo)
        else:
            raise ValueError('Formato no soportado. Solo se aceptan archivos .csv o .xlsx')
        return df
    else:
        return None

# agregar configuracion de cabezera
st.set_page_config(page_title='Trabajo Practico'
                   ,page_icon=':shark:'
                   ,layout='wide')

st.sidebar.subheader('Carga de datos')
archivo = st.sidebar.file_uploader('Seleccione un archivo', type=['csv', 'xlsx'])

# Solo cargar el carchivo si aun no existe df en st.session_state
if archivo and 'df' not in st.session_state:
    df = cargar_datos(archivo)
    st.session_state.df = df
    st.info('El archivo se ha cargado correctamente')
    st.write('El dataset tiene', df.shape[0], 'filas y', df.shape[1], 'columnas')
elif 'df' in st.session_state:
    df = st.session_state.df
else:
    st.warning('No se ha cargado ningun archivo')


opciones = ['Analisis Exploratorio',
            'JERARQUICO']

opcion = st.sidebar.radio('Selecciona una Opcion',opciones)

if opcion == 'Analisis Exploratorio':
    st.title('Analisis Exploratorio')

    #verificar si el archivo ya fue cargado
    if 'df' in st.session_state:
        df = st.session_state.df #Usamos el Datagrame almacenado en la sesion

        st.subheader('Dataframe cargado:')
        st.write(df)
        st.subheader('Primeras 5 filas')
        st.write(df.head())

        st.subheader('Informacion del Dataset')
        st.write(df.describe())

        # verificar valores nulos
        st.subheader('Valores Nulos')
        if df.isnull().sum().sum() > 0:
            st.warning('El dataset tiene valores nulos')
            st.write(df.isnull().sum())
            limpiar_datos = st.radio('Desea limpiar los datos nulos', ['Si', 'No'])


            if limpiar_datos == 'Si':
                if st.button('Confirmar'):
                    with st.spinner('Eliminando valores nulos'):
                        #Eliminar valores nulos
                        df = df.dropna().copy()
                        st.session_state.df = df
                        st.success('Valores nulos eliminados')
                        st.write(df.isnull().sum())
                        st.write("DataFrame después de eliminar los nulos:")
                        st.write(df)
            else:
                st.info('No se han eliminado los valores nulos')
        else:
            st.write(df.isnull().sum())
            st.success('El dataset no tiene valores nulos')


        # Eliminar columnas categoricas
        st.subheader('Eliminar Columnas Categoricas')
        lista_columnas = df.columns
        columnas = st.multiselect('Seleccione las columnas a eliminar', lista_columnas)


        if st.button('Eliminar Columnas'):
            with st.spinner('Eliminando columnas'):
                df = df.drop(columns=columnas).copy()
                st.session_state.df = df
                st.success('Columnas eliminadas')
                st.write("DataFrame después de eliminar columnas:")
                st.write(df)
        else:
            st.write(df)

        #Normalizacion de Datos
        st.subheader('Normalizacion de Datos')
        normalizar = st.radio('¿Desea normalizar los datos?', ['Si', 'No'])

        if normalizar == 'Si':
            metodo_normalizacion = st.radio('Seleccione el método de normalización', ['StandardScaler', 'MinMaxScaler'])

            if st.button('Normalizar Datos'):
                with st.spinner('Normalizando datos'):
                    if metodo_normalizacion == 'StandardScaler':
                        scaler = StandardScaler()

                    elif metodo_normalizacion == 'MinMaxScaler':
                        scaler = MinMaxScaler()

                    df_normalizado = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=np.number)), 
                                                    columns=df.select_dtypes(include=np.number).columns)
                    st.session_state.df_normalizado = df_normalizado
                    st.success(f'Los datos han sido normalizados con {metodo_normalizacion}')
                    st.write(df_normalizado.head())
        else:
            st.info('No se han normalizado los datos')
            st.session_state.df = df


elif opcion == 'JERARQUICO':
    st.header("Clustering - JERARQUICO")
    st.write('En esta sección se realizará el clustering jerárquico de los datos cargados')
    
    if 'df_normalizado' in st.session_state:

        # Seleccionar las columnas para el clustering
        st.sidebar.subheader('Seleccionar Columnas')
        df_normalizado = st.session_state.df_normalizado

        # Agregar columnas al multiselect
        lista_columnas = df_normalizado.columns
        columnas = st.sidebar.multiselect('Selecciona las columnas para el clustering', lista_columnas)
        
        if columnas:
                X = df_normalizado[columnas]
                st.write(df_normalizado.head())

                # Seleccionar el tipo de enlace
                enlace = st.sidebar.selectbox('Selecciona un tipo de enlace:', ['single', 'complete', 'average', 'ward'])

                # Calcular la matriz de enlace
                Z = linkage(X, enlace)
                st.write(Z)

                # Graficas el dendrograma
                fig = plt.figure(figsize=(6, 6))
                # Agregar linea de corte
                corte = st.sidebar.slider('Seleccione un valor para el corte', 0, 10, 3)

                dendrogram(Z)
                plt.axhline(y=corte, color='r', linestyle='--')
                st.pyplot(fig)

                # Crear lista de criterios
                criterios = ['maxclust', 'distance']
                criterio = st.sidebar.radio('Seleccione un criterio', criterios)
                k = st.sidebar.slider('Seleccione el numero de clusters', 2, 10, 2)

                # Asignar clusters
                clusters = fcluster(Z, k, criterion=criterio)
                df_normalizado['cluster'] = clusters
                st.write(df_normalizado.head())

                # Graficar los clusters
                fig = plt.figure(figsize=(6, 6))
                sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], c=clusters, markers='8', palette='tab10')
                st.pyplot(fig)
        else:
            st.warning('Por favor, selecciona al menos una columna')

    elif 'df' in st.session_state:

        # Seleccionar las columnas para el clustering
        st.sidebar.subheader('Seleccionar Columnas')
        df = st.session_state.df

        # Agregar columnas al multiselect
        lista_columnas = df.columns
        columnas = st.sidebar.multiselect('Selecciona las columnas para el clustering', lista_columnas)
        
        if columnas:
                X = df[columnas]
                st.write(df.head())

                # Seleccionar el tipo de enlace
                enlace = st.sidebar.selectbox('Selecciona un tipo de enlace:', ['single', 'complete', 'average', 'ward'])

                # Calcular la matriz de enlace
                Z = linkage(X, enlace)
                st.write(Z)

                # Graficas el dendrograma
                fig = plt.figure(figsize=(6, 6))
                # Agregar linea de corte
                corte = st.sidebar.slider('Seleccione un valor para el corte', 0, 10, 3)

                dendrogram(Z)
                plt.axhline(y=corte, color='r', linestyle='--')
                st.pyplot(fig)

                # Crear lista de criterios
                criterios = ['maxclust', 'distance']
                criterio = st.sidebar.radio('Seleccione un criterio', criterios)
                k = st.sidebar.slider('Seleccione el numero de clusters', 2, 10, 2)

                # Asignar clusters
                clusters = fcluster(Z, k, criterion=criterio)
                df['cluster'] = clusters
                st.write(df.head())

                # Graficar los clusters
                fig = plt.figure(figsize=(6, 6))
                sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], c=clusters, markers='8', palette='tab10')
                st.pyplot(fig)
        else:
            st.warning('Por favor, selecciona al menos una columna')