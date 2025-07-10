
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
ID_COL_NAME = 'ID_CLIENTS'

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


# --- Étape 5: Modèle Prédictif (Conditionnel) ---
# Affichez cette étape si les indicateurs sont prétraités
if st.session_state.indicators_preprocessed:
    st.header("5. Modèle Prédictif")
    st.info("Cliquez sur 'Lancer la prédiction' pour exécuter le modèle.")
    
    if st.session_state.model is None:
        st.warning("Impossible d'exécuter le modèle prédictif : le modèle n'a pas pu être chargé. Veuillez vérifier le fichier 'Model_pred.joblib'.")
    elif st.button("Lancer la prédiction", key="predict_button"):
        with st.spinner("Exécution du modèle de prédiction..."):
            # Préparer X_predict : Exclure ID_COL_NAME si elle est numérique
            # Copie de preprocessed_indicators_df pour ne pas modifier l'original.
            X_predict = st.session_state.preprocessed_indicators_df.copy()
            if ID_COL_NAME in X_predict.columns:
                X_predict = X_predict.drop(columns=[ID_COL_NAME])

            try:
                predictions = st.session_state.model.predict(X_predict)
                # Si votre modèle supporte predict_proba (pour la classification)
                probabilities = st.session_state.model.predict_proba(X_predict)[:, 1]

                st.success("Prédiction terminée ! ✅")
                st.subheader("Résultats de la Prédiction")
                
                # Base sur les indicateurs non-prétraités pour les résultats finaux
                results_df = st.session_state.indicators_df.copy() 
                results_df['Prediction_Inactivite'] = predictions
                results_df['Probabilite_Inactivite'] = probabilities
                results_df['Segment_Risque'] = results_df['Probabilite_Inactivite'].apply(segment_risk)
                
                # La colonne ID_CLIENTS est déjà dans results_df (car indicators_df la contient)
                # Nous nous assurons juste qu'elle est bien à la première position si elle existe.
                if ID_COL_NAME in results_df.columns:
                    # Si elle n'est pas déjà la première colonne, la déplacer
                    if results_df.columns.get_loc(ID_COL_NAME) != 0:
                        id_col = results_df.pop(ID_COL_NAME) # Retirer la colonne
                        results_df.insert(0, ID_COL_NAME, id_col) # Insérer au début
                else:
                    st.warning(f"La colonne d'identification '{ID_COL_NAME}' n'a pas été trouvée dans les résultats. Les ID ne seront pas affichés.")

                st.dataframe(results_df.head()) # Afficher les premières lignes des résultats

                # Statistiques de churn agrégées
                churn_count = results_df['Prediction_Inactivite'].sum()
                total_customers = len(results_df)
                churn_percentage = (churn_count / total_customers) * 100 if total_customers > 0 else 0

                st.write(f"Nombre total de clients analysés : **{total_customers}**")
                st.write(f"Nombre de clients prédits comme 'churn' : **{churn_count}**")
                st.write(f"Pourcentage de 'churn' prédit : **{churn_percentage:.2f}%**")

                # Vous pouvez ajouter plus de visualisations ici (ex: histogramme des probabilités, graphique de churn par segment)

            except Exception as e:
                st.error(f"Erreur lors de l'exécution du modèle de prédiction : {e}")
                st.warning("Assurez-vous que les données d'entrée (`X_predict`) sont dans le bon format et contiennent les bonnes colonnes pour votre modèle. Le modèle pourrait ne pas être compatible avec les données après ce prétraitement dynamique.")

#Bouton télécharger les résultats
if 'results_df' in locals():
    st.download_button(
        label="📥 Télécharger les résultats au format CSV",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name="resultats_churn.csv",
        mime='text/csv'
    )
else:
    st.info("⚠️ Les résultats ne sont pas encore disponibles. Lancez d'abord la prédiction.")
