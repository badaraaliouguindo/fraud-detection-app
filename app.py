import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Charger le modèle
model = joblib.load("fraud_detection_model.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Titre principal
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555555;'>Entrez les caractéristiques de la transaction pour prédire si elle est frauduleuse.</p>", unsafe_allow_html=True)

# Barre latérale pour les inputs
st.sidebar.header("Caractéristiques de la transaction")

feature_names = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
    "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

inputs = []

for feature in feature_names:
    # Sliders pour certaines features
    if feature in ["Time", "Amount"]:
        value = st.sidebar.slider(feature, min_value=0.0, max_value=5000.0, value=0.0, step=0.1)
    else:
        value = st.sidebar.number_input(feature, value=0.0, format="%.5f")
    inputs.append(value)

# Bouton de prédiction
if st.sidebar.button("Prédire"):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    # Affichage du résultat
    if prediction == 1:
        st.markdown(f"<h2 style='color: red;'>FRAUDE détectée ! Probabilité : {proba:.4f}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color: green;'>Transaction normale. Probabilité de fraude : {proba:.4f}</h2>", unsafe_allow_html=True)

    # Graphique de probabilité
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([0], [proba], color='red' if prediction==1 else 'green')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Probabilité de fraude")
    st.pyplot(fig)
