import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

st.title("🔢 Clasificador de Dígitos MNIST")
st.write("Entrena modelos de Machine Learning para reconocer números escritos a mano.")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Cargamos una versión reducida para mayor velocidad en el despliegue
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data[:10000], mnist.target[:10000] # Usamos 10k muestras
    return X / 255.0, y

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BARRA LATERAL (OPCIONES) ---
st.sidebar.header("Configuración del Modelo")
model_type = st.sidebar.selectbox(
    "Selecciona el Modelo",
    ("Support Vector Machine (SVM)", "Random Forest", "Logistic Regression")
)

# --- ENTRENAMIENTO ---
@st.cache_resource
def train_model(model_choice):
    if model_choice == "Support Vector Machine (SVM)":
        clf = SVC(probability=True)
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100)
    else:
        clf = LogisticRegression(max_iter=100)
    
    clf.fit(X_train, y_train)
    return clf

model = train_model(model_type)

# --- MÉTRICAS ---
st.header(f"📊 Desempeño: {model_type}")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Accuracy (Precisión)", f"{acc:.2%}")

with st.expander("Ver Reporte de Clasificación Detallado"):
    st.text(classification_report(y_test, y_pred))

# --- SECCIÓN DE PRUEBAS ---
st.divider()
st.header("🔍 Probar el Modelo")

col_test1, col_test2 = st.columns(2)

with col_test1:
    idx = st.slider("Selecciona un índice de la base de datos de prueba:", 0, len(X_test)-1, 50)
    sample_image = X_test[idx].reshape(28, 28)
    
    fig, ax = plt.subplots()
    ax.imshow(sample_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

with col_test2:
    prediction = model.predict([X_test[idx]])
    st.subheader("Resultado de la Predicción:")
    st.info(f"El modelo clasifica esta imagen como un: **{prediction[0]}**")
    
    # Probabilidades
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([X_test[idx]])
        prob_df = pd.DataFrame(probs, columns=range(10))
        st.bar_chart(prob_df.T)

st.success("¡Modelo cargado y listo para validación!")
