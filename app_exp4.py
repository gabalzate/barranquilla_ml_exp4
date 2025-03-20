import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis Electoral - Barranquilla",
    page_icon="📊",
    layout="wide"
)

# Título y descripción
st.title("📊 Dashboard de Análisis Electoral - Barranquilla")
st.markdown("""
Esta aplicación visualiza los resultados del análisis predictivo de porcentajes de votación
en Barranquilla basado en variables como ideología política, edad, años de trayectoria y resultados de encuestas.
""")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("./out/checkpoint2_barranquilla_exp4.csv")

# Cargar modelo
@st.cache_resource
def load_model():
    with open("modelo_lineal_ridge_barranquilla_exp4.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Cargar datos y modelo
try:
    df = load_data()
    model = load_model()
    st.success("Datos y modelo cargados correctamente")
except Exception as e:
    st.error(f"Error al cargar datos o modelo: {e}")
    st.stop()

# Sidebar con opciones
st.sidebar.header("Opciones de Visualización")
visualization_option = st.sidebar.selectbox(
    "Seleccionar Visualización",
    ["Resumen de Datos", "Análisis Exploratorio", "Resultados del Modelo", "Predicción Interactiva"]
)

# Función para preprocesar datos (similar a lo que hiciste en tu notebook)
def preprocess_data(dataframe):
    df_processed = dataframe.copy()
    
    # Convertir variables categóricas
    categorical_features = ['IDEOLOGÍA', 'EDAD']
    df_processed[categorical_features] = df_processed[categorical_features].astype('category')
    
    # Codificación One-Hot
    df_encoded = pd.get_dummies(df_processed, columns=categorical_features)
    
    return df_encoded

# Función para escalar características numéricas
def scale_features(dataframe):
    numerical_columns = ['ENCUESTA', 'AÑOS_TRAYECTORIA']
    scaler = StandardScaler()
    df_scaled = dataframe.copy()
    df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
    return df_scaled, scaler

# Preprocesar datos
df_encoded = preprocess_data(df)
X = df_encoded.drop(columns=['VOTOS_PORCENTAJE'])
y = df_encoded['VOTOS_PORCENTAJE']
X_scaled, scaler = scale_features(X)

# Contenido principal según la opción seleccionada
if visualization_option == "Resumen de Datos":
    st.header("Resumen del Conjunto de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primeras filas del dataset")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df.describe())
    
    st.subheader("Distribución de variables categóricas")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(df['IDEOLOGÍA'].value_counts().reset_index(), 
                     x='IDEOLOGÍA', y='count', 
                     title="Distribución por Ideología")
        st.plotly_chart(fig1)
    
    with col2:
        fig2 = px.bar(df['EDAD'].value_counts().reset_index(), 
                     x='EDAD', y='count', 
                     title="Distribución por Edad")
        st.plotly_chart(fig2)

elif visualization_option == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Tendencias"])
    
    with tab1:
        st.subheader("Distribución de Variables Numéricas")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='VOTOS_PORCENTAJE', nbins=10, 
                              title="Distribución de Porcentaje de Votos")
            st.plotly_chart(fig1)
        
        with col2:
            fig2 = px.histogram(df, x='ENCUESTA', nbins=10, 
                              title="Distribución de Resultados de Encuesta")
            st.plotly_chart(fig2)
    
    with tab2:
        st.subheader("Matriz de Correlación")
        corr = df[['VOTOS_PORCENTAJE', 'AÑOS_TRAYECTORIA', 'ENCUESTA']].corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                      color_continuous_scale='RdBu_r',
                      title="Matriz de Correlación")
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Tendencias por Variables Categóricas")
        
        fig1 = px.box(df, x='IDEOLOGÍA', y='VOTOS_PORCENTAJE', 
                    title="Porcentaje de Votos por Ideología")
        st.plotly_chart(fig1)
        
        fig2 = px.box(df, x='EDAD', y='VOTOS_PORCENTAJE', 
                    title="Porcentaje de Votos por Grupo de Edad")
        st.plotly_chart(fig2)

elif visualization_option == "Resultados del Modelo":
    st.header("Resultados del Modelo Ridge")
    
    # Hacer predicciones con el modelo
    y_pred = model.predict(X_scaled)
    
    # Crear dataframe con resultados
    results_df = pd.DataFrame({
        'Real': y,
        'Predicción': y_pred.flatten(),
        'Ideología': df['IDEOLOGÍA'],
        'Edad': df['EDAD']
    })
    
    # Mostrar coeficientes del modelo
    st.subheader("Importancia de las Características")
    coef_df = pd.DataFrame({
        'Característica': X.columns,
        'Coeficiente': model.coef_[0]
    }).sort_values('Coeficiente', key=abs, ascending=False)
    
    fig1 = px.bar(coef_df, x='Coeficiente', y='Característica', 
                orientation='h', title="Coeficientes del Modelo")
    st.plotly_chart(fig1)
    
    # Gráfico de dispersión: real vs predicción
    st.subheader("Valores Reales vs. Predicciones")
    fig2 = px.scatter(results_df, x='Real', y='Predicción', 
                    color='Ideología', symbol='Edad',
                    title="Comparación entre Valores Reales y Predicciones")
    
    # Agregar línea de referencia
    min_val = min(results_df['Real'].min(), results_df['Predicción'].min())
    max_val = max(results_df['Real'].max(), results_df['Predicción'].max())
    fig2.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                 line=dict(color='red', dash='dash'))
    
    st.plotly_chart(fig2)
    
    # Métricas del modelo
    error = results_df['Real'] - results_df['Predicción']
    mse = (error ** 2).mean()
    rmse = np.sqrt(mse)
    mae = abs(error).mean()
    r2 = 1 - sum(error ** 2) / sum((results_df['Real'] - results_df['Real'].mean()) ** 2)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Error Cuadrático Medio", f"{mse:.4f}")
    col2.metric("Raíz del Error Cuadrático Medio", f"{rmse:.4f}")
    col3.metric("Error Absoluto Medio", f"{mae:.4f}")
    col4.metric("Coeficiente R²", f"{r2:.4f}")

elif visualization_option == "Predicción Interactiva":
    st.header("Predicción Interactiva")
    st.info("Ajusta los parámetros para ver cómo afectarían el porcentaje de votos según el modelo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ideologia = st.selectbox("Ideología", df['IDEOLOGÍA'].unique())
        edad = st.selectbox("Grupo de Edad", df['EDAD'].unique())
        
    with col2:
        años_trayectoria = st.slider("Años de Trayectoria", 
                                    int(df['AÑOS_TRAYECTORIA'].min()), 
                                    int(df['AÑOS_TRAYECTORIA'].max()),
                                    int(df['AÑOS_TRAYECTORIA'].mean()))
        
        encuesta = st.slider("Resultado en Encuesta (%)", 
                           float(df['ENCUESTA'].min()), 
                           float(df['ENCUESTA'].max()),
                           float(df['ENCUESTA'].mean()))
    
    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame({
        'IDEOLOGÍA': [ideologia],
        'EDAD': [edad],
        'AÑOS_TRAYECTORIA': [años_trayectoria],
        'ENCUESTA': [encuesta]
    })
    
    # Preprocesar los datos de entrada
    input_encoded = preprocess_data(input_data)
    
    # Asegurar que tiene las mismas columnas que X (puede faltar algunas columnas dummy)
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[X.columns]
    
    # Escalar los datos de entrada
    input_scaled = input_encoded.copy()
    numerical_columns = ['ENCUESTA', 'AÑOS_TRAYECTORIA']
    
    # Usar el mismo escalador que se usó para los datos de entrenamiento
    scaler_prediction = StandardScaler()
    scaler_prediction.fit(X[numerical_columns])
    input_scaled[numerical_columns] = scaler_prediction.transform(input_scaled[numerical_columns])
    
    # Hacer la predicción
    prediction_result = model.predict(input_scaled)
    if isinstance(prediction_result, np.ndarray):
        if prediction_result.ndim > 1:
            prediction = prediction_result[0][0]
        else:
            prediction = prediction_result[0]
    else:
        prediction = prediction_result  # Si es un escalar
    
    # Mostrar la predicción
    st.subheader("Resultado de la Predicción")
    st.markdown(f"""
    ### Porcentaje de Votos Predicho: <span style='color:blue; font-size:24px'>{prediction:.2f}%</span>
    """, unsafe_allow_html=True)
    
    # Contextualizar la predicción
    avg_votes = df['VOTOS_PORCENTAJE'].mean()
    max_votes = df['VOTOS_PORCENTAJE'].max()
    
    if prediction > max_votes:
        st.success(f"¡Esta combinación predice un resultado excepcional! Por encima del máximo observado ({max_votes:.2f}%).")
    elif prediction > avg_votes:
        st.info(f"Esta combinación predice un resultado por encima del promedio ({avg_votes:.2f}%).")
    else:
        st.warning(f"Esta combinación predice un resultado por debajo del promedio ({avg_votes:.2f}%).")

# Footer
st.markdown("---")
st.markdown("Análisis Electoral - Dashboard desarrollado con Streamlit")