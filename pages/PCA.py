import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importar las librerias para el PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
if archivo:
    df = cargar_datos(archivo)
    st.session_state.df = df
    st.info('El archivo se ha cargado correctamente')
    st.write('El dataset tiene', df.shape[0], 'filas y', df.shape[1], 'columnas')
else:
    st.warning('No se ha cargado ningun archivo')


opciones = ['Analisis Exploratorio',
            'Analisis de Componentes Principales',
            'PCA - Pinguinos']

# agregar radio
opcion = st.sidebar.radio('Selecciona una Opcion',opciones)

if opcion == 'Analisis Exploratorio':
    st.title('Analisis Exploratorio')
    if 'df' in st.session_state:
        df = st.session_state.df
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
                        df = df.dropna()
                        st.success('Valores nulos eliminados')
                        st.write(df.isnull().sum())
                    st.session_state.df = df
            else:
                st.info('No se han eliminado los valores nulos')
        else:
            st.success('El dataset no tiene valores nulos')
        # Eliminar columnas categoricas
        st.subheader('Eliminar Columnas')
        df2 = st.session_state.df
        lista_columnas = df2.columns
        columnas = st.multiselect('Seleccione las columnas a eliminar', lista_columnas)
        if st.button('Eliminar Columnas'):
            with st.spinner('Eliminando columnas'):
                df2 = df2.drop(columns=columnas)
                st.success('Columnas eliminadas')
                st.write(df2)
            st.session_state.df = df2

    else:
        st.warning('No se ha cargado ningun archivo')