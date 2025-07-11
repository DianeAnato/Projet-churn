import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer



def nettoyer_et_preparer(data: pd.DataFrame) -> pd.DataFrame:
    """Renommer colonnes, imputer valeurs manquantes etc."""
    data=data.rename(columns={
               "Classe d'age": "classe_age"
    })
    data.fillna(0, inplace=True)
    data.drop_duplicates(inplace=True)
    return data
# definition tendance
    
def calculer_tendance_generique(row, colonnes, mois_num):
    valeurs = row[colonnes].values
    if np.any(pd.isnull(valeurs)):
         return np.nan
    model = LinearRegression()
    model.fit(mois_num, valeurs)
    return model.coef_[0]

def calculer_indicateurs(data: pd.DataFrame) -> pd.DataFrame:
    
    df=data.copy()
    mois_num = np.array([0, 1, 2, 3]).reshape(-1, 1)

    #Ancienneté
    mois=365.25/30
    data["Ancienneté"]=data["Age"]/mois

        # Liste des colonnes de volume par mois
    vol_novembre = [col for col in data.columns if col.startswith('N') and col.endswith('_Vol')]
    vol_decembre = [col for col in data.columns if col.startswith('D') and col.endswith('_Vol')]
    vol_janvier  = [col for col in data.columns if col.startswith('J') and col.endswith('_Vol')]
    vol_fevrier  = [col for col in data.columns if col.startswith('F') and col.endswith('_Vol')]

        # Créer les colonnes de total mensuel de volume
    data['Total_Vol_Novembre'] = data[vol_novembre].sum(axis=1)
    data['Total_Vol_Decembre'] = data[vol_decembre].sum(axis=1) 
    data['Total_Vol_Janvier']  = data[vol_janvier].sum(axis=1)
    data['Total_Vol_Fevrier']  = data[vol_fevrier].sum(axis=1)

    
    # Calcul des variations absolues
    data["Var_Abs_Decembre"] = data['Total_Vol_Decembre'] - data['Total_Vol_Novembre']
    data['Var_Abs_Janvier']  = data['Total_Vol_Janvier']  - data['Total_Vol_Novembre']
    data['Var_Abs_Fevrier']  = data['Total_Vol_Fevrier']  - data['Total_Vol_Novembre']

    # Calcul des variations relatives avec protection contre division par zéro
    for mois in ['Decembre', 'Janvier', 'Fevrier']:
        var_abs = f'Var_Abs_{mois}'
        var_rel = f'Var_Rel_{mois}'

        data[var_rel] = np.where(
        data['Total_Vol_Novembre'] != 0,
        (data[var_abs] / data['Total_Vol_Novembre']) * 100,
        np.nan
        )
    
        # Remplacer inf, -inf et NaN par 50 (ou une autre valeur si préférable)
        data[var_rel] = data[var_rel].replace([np.inf, -np.inf], np.nan).fillna(50)

    data['tendance_volume'] = data.apply(lambda row: calculer_tendance_generique(
    row, ['Total_Vol_Novembre', 'Total_Vol_Decembre', 'Total_Vol_Janvier', 'Total_Vol_Fevrier'], mois_num), axis=1)


        #Liste des colonnes de valeurs par mois
    val_novembre = [col for col in data.columns if col.startswith('N') and col.endswith('_Val')]
    val_decembre = [col for col in data.columns if col.startswith('D') and col.endswith('_Val')]
    val_janvier  = [col for col in data.columns if col.startswith('J') and col.endswith('_Val')]
    val_fevrier  = [col for col in data.columns if col.startswith('F') and col.endswith('_Val')]

    # Créer les colonnes de total mensuel de volume
    data['Total_Val_Novembre'] = data[val_novembre].sum(axis=1)
    data['Total_Val_Decembre'] = data[val_decembre].sum(axis=1)
    data['Total_Val_Janvier']  = data[val_janvier].sum(axis=1)
    data['Total_Val_Fevrier']  = data[val_fevrier].sum(axis=1)

    # tendance valeur
    
    data['tendance_valeur'] = data.apply(lambda row: calculer_tendance_generique(
    row, ['Total_Val_Novembre', 'Total_Val_Decembre', 'Total_Val_Janvier', 'Total_Val_Fevrier'], mois_num), axis=1)

    # Liste des types de transaction de dépense
    depense_types = [
        'ACTIVE_FORF', 'AIRP', 'APPCASH', 'AUTOCOLLECT', 'BILL', 'BILLSBEE', 'BILLSBEEP', 'CAGNT', 'CAGNT6',
        'CASH', 'DEBIT', 'OTHR', 'PUSH', 'USSDCASH', 'W2BCC', 'W2BCO', 'W2BE', 'W2BIIC', 'W2B0', 'W2BOA', 'W2BU',
        'XCASH', 'XMCASH'
    ]

    # Fonction pour récupérer les colonnes correspondant à un mois donné
    def get_columns_by_prefix_and_suffix(prefix, types, suffix='_Vol'):
        return [col for col in data.columns if col.startswith(prefix) and any(t in col for t in types) and col.endswith(suffix)]

    # Colonnes volume de dépense par mois
    vol_nov = get_columns_by_prefix_and_suffix('N', depense_types)
    vol_dec = get_columns_by_prefix_and_suffix('D', depense_types)
    vol_jan = get_columns_by_prefix_and_suffix('J', depense_types)
    vol_fev = get_columns_by_prefix_and_suffix('F', depense_types)

    data['Total_Depense_Nov'] = data[vol_nov].sum(axis=1)
    data['Total_Depense_Dec'] = data[vol_dec].sum(axis=1)
    data['Total_Depense_Jan'] = data[vol_jan].sum(axis=1)
    data['Total_Depense_Fev'] = data[vol_fev].sum(axis=1)


    # Appliquer la fonction à chaque ligne (individu)
    data['tendance_debit'] = data.apply(lambda row: calculer_tendance_generique(
    row, ['Total_Depense_Nov', 'Total_Depense_Dec', 'Total_Depense_Jan', 'Total_Depense_Fev'], mois_num), axis=1)

    credit_types = [
        'B2W', 'B2WE', 'B2WO', 'B2WOA', 'B2WCC', 'B2WU', 'B2WIIC', 'B2WCO', 'B2WR',
        'BATCH', 'CASH', 'CASH6', 'CSIN', 'MERCT', 'MERCT2', 'USDDCASH', 'WCASH', 'XCASH'
    ]

    vol_credit_nov = get_columns_by_prefix_and_suffix('N', credit_types)
    vol_credit_dec = get_columns_by_prefix_and_suffix('D', credit_types)
    vol_credit_jan = get_columns_by_prefix_and_suffix('J', credit_types)
    vol_credit_fev = get_columns_by_prefix_and_suffix('F', credit_types)

    data['Total_Credit_Nov'] = data[vol_credit_nov].sum(axis=1)
    data['Total_Credit_Dec'] = data[vol_credit_dec].sum(axis=1)
    data['Total_Credit_Jan'] = data[vol_credit_jan].sum(axis=1)
    data['Total_Credit_Fev'] = data[vol_credit_fev].sum(axis=1)


    # Appliquer la fonction à chaque ligne (individu)
    data['tendance_credit'] = data.apply(lambda row: calculer_tendance_generique(
    row, ['Total_Credit_Nov', 'Total_Credit_Dec', 'Total_Credit_Jan', 'Total_Credit_Fev'], mois_num), axis=1)


    # Supposons que les colonnes totales de débit et crédit sont déjà calculées
    data['Ratio_Debit_Credit_Nov'] = data['Total_Depense_Nov'] / data['Total_Credit_Nov']
    data['Ratio_Debit_Credit_Dec'] = data['Total_Depense_Dec'] / data['Total_Credit_Dec']
    data['Ratio_Debit_Credit_Jan'] = data['Total_Depense_Jan'] / data['Total_Credit_Jan']
    data['Ratio_Debit_Credit_Fev'] = data['Total_Depense_Fev'] / data['Total_Credit_Fev']

        # Remplacer les NaN et inf par 0 (ou une autre valeur si nécessaire)
    data['Ratio_Debit_Credit_Nov'] = data['Ratio_Debit_Credit_Nov'].replace([float('inf'), -float('inf'), None], 0)
    data['Ratio_Debit_Credit_Dec'] = data['Ratio_Debit_Credit_Dec'].replace([float('inf'), -float('inf'), None], 0)
    data['Ratio_Debit_Credit_Jan'] = data['Ratio_Debit_Credit_Jan'].replace([float('inf'), -float('inf'), None], 0)
    data['Ratio_Debit_Credit_Fev'] = data['Ratio_Debit_Credit_Fev'].replace([float('inf'), -float('inf'), None], 0)

    # Remplacer les inf et NaN par 0 dans toutes les colonnes ratio
    ratio_cols = ['Ratio_Debit_Credit_Nov', 'Ratio_Debit_Credit_Dec', 'Ratio_Debit_Credit_Jan', 'Ratio_Debit_Credit_Fev']
    data[ratio_cols] = data[ratio_cols].replace([float('inf'), -float('inf')], np.nan)
    data[ratio_cols] = data[ratio_cols].fillna(0)

    

    # Appliquer à chaque individu
    data['tendance_ratio_dc'] =data.apply(lambda row: calculer_tendance_generique(
    row, ['Ratio_Debit_Credit_Nov', 'Ratio_Debit_Credit_Dec', 'Ratio_Debit_Credit_Jan', 'Ratio_Debit_Credit_Fev'], mois_num), axis=1)

    #RFM####"
    # Sélection des colonnes de volume et de valeur
    vol_cols = [col for col in data.columns if col.endswith('_Vol')]
    val_cols = [col for col in data.columns if col.endswith('_Val')]

    # Calcul des fréquences et montants totaux
    data['RFM_Frequence'] = data[vol_cols].sum(axis=1)
    data['RFM_Montant'] = data[val_cols].sum(axis=1)
    # Fonction pour trouver le mois de dernière activité
    def get_last_active_month(row):
        months = ['Novembre', 'Decembre', 'Janvier', 'Fevrier']
        mois_codes = ['N', 'D', 'J', 'F']
        for mois_code, mois_name in reversed(list(zip(mois_codes, months))):
            matching_cols = [col for col in data.columns if col.startswith(mois_code) and col.endswith('_Vol')]
            if row[matching_cols].sum() > 0:
                return mois_name
        return 'Aucun'

    # Appliquer la fonction
    data['RFM_Recence'] = data.apply(get_last_active_month, axis=1)
    # Dictionnaire de poids (plus récent = 0)
    recence_map = {'Fevrier': 0, 'Janvier': 1, 'Decembre': 2, 'Novembre': 3, 'Aucun': 4}
    data['RFM_Recence_Num'] = data['RFM_Recence'].map(recence_map)

        # 2. Extraire les types de transaction (sans le préfixe mois)
    # Exemple : "NRetrait_Vol" => "Retrait"
    transaction_types = set([re.sub(r'^[NDJF]', '', col).replace('_Vol', '') for col in vol_cols])

    # 3. Pour chaque type de transaction, vérifier s’il a été utilisé au moins une fois (dans n’importe quel mois)
    for t in transaction_types:
        related_cols = [col for col in vol_cols if t in col]
        data[f'{t}_used'] = data[related_cols].sum(axis=1).gt(0).astype(int)

    # 4. Compter le nombre total de types utilisés par client
    used_cols = [col for col in data.columns if col.endswith('_used')]
    data['nbre_types_utilises'] = data[used_cols].sum(axis=1)

    #Coefficient de variation
    data['CV_vol'] = data[vol_cols].std(axis=1) / (data[vol_cols].mean(axis=1) + 1e-6)
    # Calcul de l'écart type des volumes
    data['Vol_std'] = data[vol_cols].std(axis=1)


    # Sélectionnez les colonnes d'indicateurs que vous souhaitez retourner

    colonnes_1 = ['ID_CLIENTS','Ancienneté',
    'nbre_types_utilises', 'CV_vol', 'RFM_Recence_Num', 'RFM_Frequence',
    'RFM_Montant', 'tendance_ratio_dc',  
    'tendance_valeur', 'tendance_volume','tendance_credit','tendance_debit',
    'Var_Rel_Decembre','Var_Rel_Janvier','Var_Rel_Fevrier','Total_Vol_Novembre', 
    'Total_Vol_Decembre', 'Total_Vol_Janvier', 'Total_Vol_Fevrier',
    'Total_Val_Novembre', 'Total_Val_Decembre', 'Total_Val_Janvier', 'Total_Val_Fevrier',
    'Total_Credit_Nov', 'Total_Credit_Dec', 'Total_Credit_Jan', 'Total_Credit_Fev',
    'Total_Depense_Nov', 'Total_Depense_Dec', 'Total_Depense_Jan', 'Total_Depense_Fev',
    'Ratio_Debit_Credit_Nov', 'Ratio_Debit_Credit_Dec', 'Ratio_Debit_Credit_Jan', 'Ratio_Debit_Credit_Fev',
    'Vol_std']

    indicator_df=data[colonnes_1]
    
    

    # Identifier les colonnes de type transaction
    mois_prefixes = ['N', 'D', 'J', 'F']
    types_transaction = set()

    for col in df.columns:
        if col[0] in mois_prefixes:
            types_transaction.add(col[1:])  # Ex : 'PaiementVol', 'RetraitVal', etc.

    # Séparer les types en Volume et Valeur
    types_vol = {t for t in types_transaction if t.endswith('Vol')}
    types_val = {t for t in types_transaction if t.endswith('Val')}

    # Fonction pour sommer les 4 mois pour chaque type
    def somme_mensuelle(df, types, suffixe):
        for tx in types:
            colonnes = [m + tx for m in mois_prefixes if (m + tx) in data.columns]
            if colonnes:
                df[f'Somme_{tx}'] = data[colonnes].sum(axis=1)
        return df

    # Application sur la base principale
    df = somme_mensuelle(data, types_vol, 'Vol')
    df = somme_mensuelle(data, types_val, 'Val')

    Type_trans=['Somme_CSIN_Vol','Somme_CAGNT_Vol','Somme_DEBIT_Vol','Somme_PUSH_Vol',
                'Somme_MRCH_Vol','Somme_USSDCASH_Vol','Somme_AIRT_Vol',
                'Somme_MERCT_Vol','Somme_BILLSBEE_Vol','Somme_APPCASH_Vol','Somme_WCASH_Vol']
    Type2=['CSIN','CAGNT','DEBIT','PUSH','MRCH','USSDCASH','AIRT','MERCT','BILLSBEE','APPCASH','WCASH']

    indicator_df[Type2]=df[Type_trans]

    # Retourne un DataFrame avec seulement les colonnes d'indicateurs
    return indicator_df

def pretraiter_indicateurs(df_indicators: pd.DataFrame, id_col_name: str) -> pd.DataFrame:
    # ... corps de la fonction ...
    """
    Applique la standardisation (Standard Scaling) et l'écrêtage des valeurs aberrantes.
    AVERTISSEMENT : Cette méthode adapte le StandardScaler et calcule les bornes d'outliers
    sur les données fournies (les données téléchargées). Ceci n'est généralement PAS recommandé
    pour le déploiement, car le modèle a été entraîné sur des données mises à l'échelle avec des
    statistiques spécifiques et fixes provenant du jeu de données d'entraînement. Cela entraînera
    des prédictions incohérentes si les statistiques des données téléchargées diffèrent
    significativement de celles du jeu d'entraînement.
    """
    if df_indicators.empty:
        print("AVERTISSEMENT : DataFrame d'indicateurs vide. Aucun prétraitement appliqué.")
        return df_indicators.copy()
    
    # Sélectionne toutes les colonnes numériques du DataFrame d'indicateurs.
    # Ceci correspond à l'utilisation que vous avez fournie.
    numerical = df_indicators.select_dtypes(include=['number']).columns
    
    # Vérifier s'il y a des colonnes numériques à traiter
    if numerical.empty:
        print("AVERTISSEMENT : Aucun indicateur numérique trouvé pour le prétraitement.")
        return df_indicators.copy()


    # Étape 1 : Écrêtage des valeurs aberrantes (Outlier Clipping)
    # Les bornes sont calculées sur les données actuelles (déjà standardisées).
    base_corr= df_indicators.copy() # Crée une copie du DataFrame déjà standardisé pour appliquer l'écrêtage
    
    for col in numerical:
        Q1 = base_corr[col].quantile(0.25)
        Q3 = base_corr[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        base_corr[col] = base_corr[col].clip(lower=lower_bound, upper=upper_bound)

  
    print("Gestion des outliers par bornage appliquée (bornes calculées sur les données actuelles).")
    # Etape 2
    # Supposons que X est ton dataframe de variables numériques
    X = base_corr.drop(columns=['ID_CLIENTS'])
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_transformed = pt.fit_transform(X)

    # On récupère un DataFrame avec les bons noms de colonnes
    X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)

    if id_col_name in df_indicators.columns:
        # Ajouter l'ID (non transformé) comme première colonne
        id_col = df_indicators[[id_col_name]].copy()
        X_transformed_df = pd.concat([id_col, X_transformed_df], axis=1)


    return X_transformed_df

# segmentation
def segment_risk(score):
    if score < 0.3:
        return "Faible"
    elif score < 0.6:
        return "Moyen"
    else:
        return "Élevé"





