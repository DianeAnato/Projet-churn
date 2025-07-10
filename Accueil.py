import pandas as pd
import numpy as np
import joblib
import os 
#from xgboost import XGBClassifier
#from sklearn.preprocessing import StandardScaler
#import seaborn as sns # Décommentez si vous l'utilisez plus tard

import streamlit as st

from utils import segment_risk
from utils import nettoyer_et_preparer, calculer_indicateurs,pretraiter_indicateurs

st.set_page_config(layout="wide") # Bonne pratique pour un contenu plus large
st.title("Application prédictive sur le churn")

with st.expander("ℹ️ À propos de cette application", expanded=True):
    st.markdown("""
    Cette application interactive a pour objectif de prédire le risque d'inactivité des clients Moov Money.

    Elle repose sur un modèle de machine learning entraîné à partir des comportements transactionnels des clients et permet de :
    
    - Nettoyer et transformer les données client,
    - Construire des indicateurs clés (RFM, tendances, fréquences, etc.),
    - Générer des prédictions de probabilité d'inactivité,
    - Segmenter les clients selon leur niveau de risque (Faible, Moyen, Élevé),
    - Visualiser les résultats et extraire des tableaux exportables.

   Dans la base de données à insérer il faut impérativement les données suivantes :
    - ID_CLIENTS : Identifiant unique du client (obligatoire pour lier les résultats),
    -  Age du compte client,
    - Les types de transaction :'ACTIVE_FORF', 'AIRP', 'APPCASH', 'AUTOCOLLECT', 'BILL', 'BILLSBEE', 'BILLSBEEP', 'CAGNT', 'CAGNT6',
        'CASH', 'DEBIT', 'OTHR', 'PUSH', 'USSDCASH', 'W2BCC', 'W2BCO', 'W2BE', 'W2BIIC', 'W2B0', 'W2BOA', 'W2BU',
        'XCASH', 'XMCASH','B2W', 'B2WE', 'B2WO', 'B2WOA', 'B2WCC', 'B2WU', 'B2WIIC', 'B2WCO', 'B2WR',
        'BATCH', 'CASH', 'CASH6', 'CSIN', 'MERCT', 'MERCT2', 'USDDCASH', 'WCASH', 'XCASH'
   - Les transactions sont en volume et en valeur.
    
    **💡 Astuce** : Veillez à suivre les étapes dans l’ordre (nettoyage → indicateurs → prétraitement → prédiction) pour garantir la qualité des résultats.
    """)


# --- Initialisation de st.session_state ---
# Ceci garantit que les drapeaux sont configurés lors du premier lancement de l'application
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'indicators_calculated' not in st.session_state:
    st.session_state.indicators_calculated = False
if 'indicators_preprocessed' not in st.session_state: 
    st.session_state.indicators_preprocessed = False

# Il est judicieux de stocker les DataFrames dans session_state également,
# afin qu'ils persistent entre les réexécutions sans avoir besoin de re-télécharger ou re-traiter
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'indicators_df' not in st.session_state:
    st.session_state.indicators_df = None
if 'preprocessed_indicators_df' not in st.session_state: 
    st.session_state.preprocessed_indicators_df = None
if 'client_ids' not in st.session_state: 
    st.session_state.client_ids = None
if 'model' not in st.session_state:
    st.session_state.model = None


# # Charger le modèle DANS le bloc d'initialisation ou via une vérification

if 'client_ids' not in st.session_state: 
    st.session_state.client_ids = None
if 'model' not in st.session_state:
    st.session_state.model = None

# @st.cache_resource # Utiliser st.cache_resource pour charger le modèle une seule fois
# def load_model(): # Renommé pour éviter la confusion avec st.session_state.model
#     model_path = 'Model_pred.joblib'
    
#     # --- DIAGNOSTIC AJOUTÉ ICI ---
#     if not os.path.exists(model_path):
#         st.error(f"Erreur : Le fichier du modèle '{model_path}' N'EXISTE PAS à l'emplacement attendu. Veuillez vérifier le nom et le répertoire.")
#         # Afficher le répertoire de travail actuel pour aider au débogage du chemin
#         st.info(f"Répertoire de travail actuel : {os.getcwd()}")
#         return None
        
#     try:
#         model = joblib.load(filename=model_path)
#         st.success(f"Modèle '{model_path}' chargé avec succès.")
#         return model
#     except FileNotFoundError: # Normalement, cette erreur est gérée par os.path.exists maintenant
#         st.error(f"Erreur (FileNotFoundError) : Le fichier du modèle '{model_path}' n'a pas été trouvé. Ceci est inattendu après la vérification d'existence.")
#         return None
#     except Exception as e:
#         st.error(f"Erreur inattendue lors du chargement du modèle '{model_path}' : {e}")
#         st.info(f"Cela peut indiquer un fichier corrompu, des problèmes de permissions ou une incompatibilité de version (ex: joblib/scikit-learn).")
#         return None

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "Model_pred.joblib")

    st.write(f"🔍 Chemin complet du modèle : {model_path}")

    if not os.path.exists(model_path):
        st.error("❌ Le modèle n'a pas été trouvé.")
        return None

    try:
        model = joblib.load(model_path)
        st.success("✅ Modèle chargé avec succès.")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None


if st.session_state.model is None: 
     st.session_state.model =load_model()

# # Charger le modèle DANS le bloc d'initialisation ou via une vérification
if st.session_state.model is None: 
     st.session_state.model =load_model()


# NOUVELLE INITIALISATION REQUISE POUR CORRIGER L'ERREUR
if 'uploaded_file_name' not in st.session_state:
     st.session_state.uploaded_file_name = None 

st.markdown("""
    <style>
    body {
        background-color: #ADD8E6  ; /* fond blanc */
    }

    .block-container {
        padding: 2rem 1rem;
        background-color: #ADD8E6  ;
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

