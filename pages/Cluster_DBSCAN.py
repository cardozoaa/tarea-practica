import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


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
            'DBSCAN']

# Seleccionar opcion
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
        
        # Seleccionar las columnas para el clustering
        st.subheader('Seleccionar Columnas ejes X e Y')
        if 'df_normalizado' in st.session_state:
            df_normalizado = st.session_state.df_normalizado
            eje_x = st.selectbox("Selecciona una opción para el eje X:", df_normalizado.columns)
            eje_y = st.selectbox("Selecciona una opción para el eje Y:", df_normalizado.columns)

            if st.button('Seleccionar Columnas'):
                with st.spinner('Seleccionando'):
                    df_normalizado = df_normalizado[[eje_x, eje_y]]
                    st.session_state.df_seleccionado = df_normalizado
                    st.success('Columnas seleccionadas')
                    st.write(df_normalizado.head())

        elif 'df' in st.session_state:
            df = st.session_state.df
            eje_x = st.selectbox("Selecciona una opción para el eje X:", df.select_dtypes(include=np.number).columns)
            eje_y = st.selectbox("Selecciona una opción para el eje Y:", df.select_dtypes(include=np.number).columns)

            if st.button('Seleccionar Columnas'):
                with st.spinner('Seleccionando'):
                    df_seleccionado = df[[eje_x, eje_y]]
                    st.session_state.df_seleccionado = df_seleccionado
                    st.success('Columnas seleccionadas')
                    st.write(df_seleccionado.head())


elif opcion == 'DBSCAN':
    st.header("Clustering - DBSCAN")
    st.write('DBSCAN es un algoritmo de clustering basado en densidad que agrupa puntos en clusters de alta densidad.')

    if 'df_seleccionado' not in st.session_state:
        st.warning("Por favor, selecione los ejes X e Y en la opción de Análisis Exploratorio")
    else:
        df_seleccionado=st.session_state.df_seleccionado
    
        datos=df_seleccionado.to_numpy()

        # graficar nuestros datos

        fig= plt.figure(figsize=(6, 6))
        plt.scatter(datos[:, 0], datos[:, 1], s=100) #2 columnas

        plt.xlabel(df_seleccionado.columns[0])
        plt.ylabel(df_seleccionado.columns[1])
        plt.title('Datos de Vino')
        st.pyplot(fig)

        # min_samples
        min_samples= st.slider('min_samples', 1, 20, 3)

        # epsilon
        epsilon= st.slider('epsilon', 0.1, 10.0, 0.5)

        # lista de distancias
        lista_distancias=['euclidean', 'manhattan', 'chebyshev']
        distancia= st.selectbox('Selecciona una distancia:', lista_distancias)

        # ajustar el modelo DBSCAN
        dbscan=DBSCAN(eps=epsilon,min_samples=min_samples, metric=distancia).fit_predict(datos)

        #st.write(dbscan)
        # calcular el coeficiente de silueta
        silhouette= silhouette_score(datos, dbscan)

        # graficar los clusters
        col1,col2=st.columns(2)
        with col1:
            fig= plt.figure(figsize=(6, 6))
            plt.scatter(datos[:, 0], datos[:, 1], c=dbscan, s=100)
            plt.xlabel(df_seleccionado.columns[0])
            plt.ylabel(df_seleccionado.columns[1])
            plt.title('DBSCAN Clustering')
            st.pyplot(fig)
        with col2:
            st.markdown('## Medidas de Evaluación')
            st.write('Número de Clusters:', len(np.unique(dbscan)))
            st.write('Silhouette Score:', silhouette)