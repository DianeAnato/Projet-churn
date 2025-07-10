
import pandas as pd
import numpy as np
import joblib
import os 

import streamlit as st

from utils import segment_risk
from utils import nettoyer_et_preparer, calculer_indicateurs,pretraiter_indicateurs
st.markdown("""
    <style>
    body {
        background-color: #FFA500; /* fond blanc */
    }

    .block-container {
        padding: 2rem 1rem;
        background-color: #FFA500;
    }

    h1, h2, h3 {
        color: #007BFF; /* titres bleus */
    }

    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #0056b3;
    }

    .stAlert {
        background-color: #ffffff; /* orange clair pour infos */
        color: black;
    }

    .stDownloadButton>button {
        background-color: #ffffff;
        color: white;
        border-radius: 5px;
    }

    </style>
""", unsafe_allow_html=True)


ID_COL_NAME = 'ID_CLIENTS'

# --- Étape 1: Chargement des Données ---
st.header("1. Chargement des Données")
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

# Seulement si un nouveau fichier est téléchargé, nous le traitons
if uploaded_file is not None:
    # Vérifier si c'est un nouveau fichier ou un fichier qui a été re-téléchargé
    if st.session_state.raw_data is None or uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté. Veuillez importer un fichier CSV ou Excel.")
                st.session_state.data_uploaded = False
                st.stop()
            
            # Gestion de la colonne d'identification
            if ID_COL_NAME in data.columns:
                # Ne PAS supprimer la colonne ID de 'data'
                st.session_state.client_ids = data[ID_COL_NAME].reset_index(drop=True)
                st.session_state.raw_data = data # Garder la colonne ID dans raw_data
                st.success(f"Colonne d'identification '{ID_COL_NAME}' détectée et conservée.")
            else:
                st.warning(f"La colonne '{ID_COL_NAME}' n'a pas été trouvée. Les résultats ne seront pas liés à un identifiant client.")
                st.session_state.client_ids = None
                st.session_state.raw_data = data # Garder les données brutes telles quelles si pas d'ID
            
            st.success("Fichier chargé avec succès ✅")
            st.subheader("Aperçu des données brutes")
            st.dataframe(st.session_state.raw_data.head()) # Affiche raw_data qui contient maintenant l'ID
            
            # Mettre à jour l'état et réinitialiser les étapes suivantes
            st.session_state.data_uploaded = True 
            st.session_state.data_processed = False
            st.session_state.indicators_calculated = False
            st.session_state.indicators_preprocessed = False # Réinitialiser aussi le drapeau de prétraitement
            st.session_state.uploaded_file_name = uploaded_file.name # Garder le nom du fichier pour comparaison
            st.rerun() # FORCER LA RÉ-EXÉCUTION pour afficher l'étape suivante
            
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.session_state.data_uploaded = False
            st.session_state.raw_data = None # Réinitialiser les données en cas d'erreur
            st.session_state.client_ids = None # Réinitialiser l'ID aussi en cas d'erreur
    else:
        # Si le même fichier est "re-sélectionné", on affiche juste son aperçu
        st.success("Fichier déjà chargé ✅")
        st.subheader("Aperçu des données brutes")
        st.dataframe(st.session_state.raw_data.head())
        st.session_state.data_uploaded = True # Assurer que le drapeau est à True

elif st.session_state.data_uploaded and st.session_state.raw_data is not None:
    # Cas où le fichier a déjà été chargé et traité lors d'une session précédente
    st.success("Fichier déjà chargé ✅")
    st.subheader("Aperçu des données brutes")
    st.dataframe(st.session_state.raw_data.head())
else:
    st.info("Veuillez importer un fichier pour continuer.")


# --- Étape 2: Nettoyage et Préparation des Données ---
# Cette section n'apparaît que si les données ont été téléchargées
if st.session_state.data_uploaded:
    st.header("2. Nettoyage et Préparation des Données")
    
    if st.session_state.data_processed:
        # Si les données sont déjà traitées, on les affiche et on propose de retraiter
        st.subheader("Aperçu des données traitées (déjà traité)")
        st.dataframe(st.session_state.processed_data.head())
        st.info("Cliquez sur 'Retraiter les données' si vous souhaitez les traiter à nouveau.")
        if st.button("Retraiter les données", key="reprocess_button"):
            st.session_state.data_processed = False # Réinitialiser le drapeau pour activer le bouton "Traiter"
            st.session_state.indicators_calculated = False # Réinitialiser les étapes suivantes
            st.rerun() # FORCER LA RÉ-EXÉCUTION
    else:
        # Si les données ne sont pas encore traitées, proposer de les traiter
        st.info("Cliquez sur 'Traiter les données' pour nettoyer et préparer la base.")
        if st.button("Traiter les données", key="process_button"):
            with st.spinner("Traitement des données en cours..."):
                st.session_state.processed_data = nettoyer_et_preparer(st.session_state.raw_data.copy())
            
            st.success("Données traitées avec succès ✅")
            st.subheader("Aperçu des données traitées")
            st.dataframe(st.session_state.processed_data.head())
            st.session_state.data_processed = True # Mettre à jour le drapeau
            st.session_state.indicators_calculated = False # Réinitialiser les étapes suivantes
            st.rerun() # FORCER LA RÉ-EXÉCUTION pour afficher l'étape suivante


