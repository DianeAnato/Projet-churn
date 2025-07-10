import pandas as pd
import numpy as np

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


# --- Étape 3: Calcul des Indicateurs ---
# Cette section n'apparaît que si les données ont été traitées
if st.session_state.data_processed:
    st.header("3. Calcul des Indicateurs")

    if st.session_state.indicators_calculated:
        # Si les indicateurs sont déjà calculés, on les affiche et on propose de recalculer
        st.subheader("Indicateurs Calculés (déjà calculés)")
        st.dataframe(st.session_state.indicators_df)
        st.info("Cliquez sur 'Recalculer les indicateurs' si vous souhaitez les mettre à jour.")
        if st.button("Recalculer les indicateurs", key="recalculate_button"):
            st.session_state.indicators_calculated = False # Réinitialiser le drapeau
            st.rerun() # FORCER LA RÉ-EXÉCUTION
    else:
        # Si les indicateurs ne sont pas encore calculés, proposer de les calculer
        st.info("Cliquez sur 'Calculer les indicateurs' pour obtenir les métriques clés.")
        if st.button("Calculer les indicateurs", key="calculate_button"):
            with st.spinner("Calcul des indicateurs en cours..."):
                st.session_state.indicators_df = calculer_indicateurs(st.session_state.processed_data.copy())
            
            st.success("Indicateurs calculés avec succès ✅")
            st.subheader("Indicateurs Calculés")
            st.dataframe(st.session_state.indicators_df)
            st.session_state.indicators_calculated = True # Mettre à jour le drapeau
            st.rerun() # FORCER LA RÉ-EXÉCUTION pour afficher l'étape suivante


# ÉTAPE 4: Prétraitement des Indicateurs ---
if st.session_state.indicators_calculated:
    st.header("4. Prétraitement des Indicateurs")
    
    if st.session_state.indicators_preprocessed:
        st.subheader("Indicateurs Prétraités (déjà traités)")
        st.dataframe(st.session_state.preprocessed_indicators_df.head()) # Affiche prétraitement avec l'ID
        st.info("Cliquez sur 'Reprétraiter les indicateurs' si nécessaire.")
        if st.button("Reprétraiter les indicateurs", key="repreprocess_button"):
            st.session_state.indicators_preprocessed = False
            st.rerun()
    else:
        st.info("Cliquez sur 'Prétraiter les indicateurs' pour appliquer la standardisation et gérer les outliers.")
        if st.button("Prétraiter les indicateurs", key="preprocess_button"):
            with st.spinner("Prétraitement des indicateurs en cours (standardisation et bornage)..."):
                # --- MODIFICATION ICI : Passer ID_COL_NAME à la fonction pretraiter_indicateurs ---
                st.session_state.preprocessed_indicators_df = pretraiter_indicateurs(st.session_state.indicators_df.copy(), ID_COL_NAME)
                
                # Vérifier si la fonction a retourné un DataFrame non vide
                if st.session_state.preprocessed_indicators_df is not None and not st.session_state.preprocessed_indicators_df.empty:
                    st.success("Indicateurs prétraités avec succès ✅")
                    st.subheader("Indicateurs Prétraités (standardisés et outliers gérés)")
                    st.dataframe(st.session_state.preprocessed_indicators_df.head()) # Affiche prétraitement avec l'ID
                    st.session_state.indicators_preprocessed = True
                    st.rerun()
                else:
                    st.error("Le prétraitement des indicateurs a échoué ou a renvoyé un DataFrame vide. Vérifiez les logs de la console pour plus de détails.")
                    st.session_state.indicators_preprocessed = False # Marquer comme échoué pour ne pas avancer
