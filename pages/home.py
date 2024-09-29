import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#agregar configuracion de la pagina
st.set_page_config(page_title='TP - Maestria en Ciencia de Datos', layout='wide')

st.title('Trabajo Practico - Maestria en Ciencia de Datos')

st.sidebar.title('Menu')

#lista de opciones
opciones = ['Carga de datos', 'Analisis exploratorio']

#seleccion de opciones
opcion = st.sidebar.selectbox('Seleccione una opcion', opciones)

@st._cache_data
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



if opcion == 'Carga de datos':
    st.sidebar.subheader('Carga de datos')
    archivo = st.sidebar.file_uploader('Seleccione un archivo', type=['csv', 'xlsx'])
    if archivo:
        df = cargar_datos(archivo)
        st.session_state.df = df
        st.info('Datos cargados correctamente')
    else:
        st.write('No se ha cargado ningun archivo')
    

    