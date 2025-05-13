import streamlit as st
import joblib
import numpy as np

# Chargement du modèle et du scaler
model = joblib.load('modele.pkl')
scaler = joblib.load('scaler.pkl')

# Interface utilisateur
st.title("🔧 Simulateur de Prédiction de Panne - Industrie 4.0")
st.write("Entrez les valeurs exactes pour les paramètres suivants.")

# Entrée utilisateur
tempMode = st.number_input("TempMode (de 0 à 7)", min_value=0, max_value=7, value=0)
footfall = st.number_input("Footfall", min_value=0.0, value=100.0)
AQ = st.number_input("AQ", min_value=0.0, value=50.0)
USS = st.number_input("USS", min_value=0.0, value=50.0)
CS = st.number_input("CS", min_value=0.0, value=50.0)
VOC = st.number_input("VOC", min_value=0.0, value=50.0)
RP = st.number_input("RP", min_value=0.0, value=50.0)
IP = st.number_input("IP", min_value=0.0, value=50.0)
Temperature = st.number_input("Temperature", min_value=-50.0, value=25.0)

# Prédiction
if st.button("Prédire"):
    # Crée un tableau des données numériques à normaliser
    numeric_data = np.array([[footfall, AQ, USS, CS, VOC, RP, IP, Temperature]])
    
    # Standardisation
    scaled_data = scaler.transform(numeric_data)

    # Ajout de tempMode non standardisé au début
    final_input = np.hstack([[tempMode], scaled_data])

    # Prédiction
    prediction = model.predict(final_input.reshape(1, -1))[0]

    # Affichage du résultat
    if prediction == 1:
        st.markdown('<div style="color: red; font-weight: bold;">🛑 Panne probable</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: green; font-weight: bold;">✅ Pas de panne</div>', unsafe_allow_html=True)
