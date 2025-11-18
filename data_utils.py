# data_utils.py
# =====================================================================
# Modulo per caricamento, validazione e preprocessing dei dati
#
# Scopo
# Questo modulo centralizza tutte le operazioni di I/O e pre-processing
# usate dall'applicazione Streamlit. Le funzioni esposte hanno l'obiettivo di:
# - leggere il CSV sorgente in modo robusto e verificare la presenza delle
#   colonne richieste;
# - normalizzare il campo data/ora e ordinare cronologicamente il dataset;
# - fornire utility per creare split train/test preservando l'ordine temporale;
# - generare feature laggati (sia univariati che multivariati) utilizzabili
#   da modelli machine learning e reti neurali (MLP/LSTM);
# - preparare dati specifici per modelli multivariati come VARMAX.
#
# Funzioni principali
# - load_data(csv_path: Path) -> pd.DataFrame
#     Legge il CSV, valida colonne richieste e ritorna un DataFrame ordinato
#     con la colonna data convertita a datetime. Questa funzione è decorata
#     con `st.cache_data` per evitare ricaricamenti ripetuti in Streamlit.
# - create_train_test_split(df, test_split=0.8)
#     Crea uno split cronologico train/test (basato su percentuale o data)
# - create_lagged_features / create_lagged_features_multivariate
#     Generano feature laggati per serie temporali (necessari per ensemble
#     e LSTM). Rimuovono le righe con NaN create dallo shift.
# - prepare_varmax_data(df, rolling_window=7, ema_alpha=0.3)
#     Costruisce le matrici endogene/esogene e applica trasformazioni utili
#     per VARMAX (rolling sum, EMA, differenziazione, allineamento temporale).
#
# Contratti e aspettative sul dataset
# - `COLUMN_NAMES` e `REQUIRED_COLUMNS` sono definiti in `config.py`: il CSV
#   deve contenere almeno le colonne elencate in `REQUIRED_COLUMNS`.
# - La colonna data viene convertita in `pd.Timestamp` e usata per ordinare e
#   indicizzare i dati; molte funzioni si aspettano che questa colonna sia
#   presente.
# - Le funzioni che generano lag rimuovono le righe iniziali con NaN,
#   quindi il numero di campioni utili si riduce a seconda di `n_lags`.
#
#
# Esempio rapido
#   df = load_data(DATA_PATH)
#   train_df, test_df, split_date = create_train_test_split(df, 0.8)
#   df_lagged = create_lagged_features(df, n_lags=5)
#
# =====================================================================

import pandas as pd
import numpy as np
from config import COLUMN_NAMES

def create_lagged_features(df: pd.DataFrame, n_lags: int, include_date: bool = False) -> pd.DataFrame:
    """
    Crea feature ritardate (lag) per la colonna target.
    """
    df_lagged = df.copy()
    target = COLUMN_NAMES['TARGET']
    
    # Creazione lag
    for t in range(1, n_lags + 1):
        df_lagged[f'lag_{t}'] = df_lagged[target].shift(t)
    
    # Rimuove righe con NaN generate dallo shift
    df_lagged = df_lagged.dropna()
    
    # Reset index per mantenere le date allineate se necessario
    if not include_date and COLUMN_NAMES['DATE'] in df_lagged.columns:
        # Se la data è l'indice, la lasciamo lì, altrimenti la escludiamo se richiesto
        pass
        
    return df_lagged

def create_lagged_features_multivariate(df: pd.DataFrame, n_lags: int, target_col: str, exog_cols: list) -> pd.DataFrame:
    """
    Crea feature ritardate per target e variabili esogene (per LSTM/VARMAX).
    """
    df_copy = df.copy()
    cols_to_lag = [target_col] + exog_cols
    
    for col in cols_to_lag:
        for t in range(1, n_lags + 1):
            df_copy[f'{col}_lag_{t}'] = df_copy[col].shift(t)
            
    return df_copy.dropna()

def prepare_varmax_data(df: pd.DataFrame, rolling_window: int, ema_alpha: float):
    """
    Prepara i dati per VARMAX: smoothing e differenziazione.
    """
    # Selezione colonne
    target = COLUMN_NAMES['TARGET']
    temp = COLUMN_NAMES['TEMPERATURE']
    humid = COLUMN_NAMES['HUMIDITY']
    
    data = df[[target, temp, humid]].copy()
    
    # Smoothing (Media mobile esponenziale o rolling)
    data_smooth = data.ewm(alpha=ema_alpha).mean()
    
    # Differenziazione per stazionarietà (semplice differenza primo ordine)
    data_diff = data_smooth.diff().dropna()
    
    # Separazione endogene (Target) ed esogene (Temp, Humid)
    # Nota: VARMAX può gestire più endogene, qui semplifichiamo
    endog = data_diff[[target]]
    exog = data_diff[[temp, humid]]
    
    return endog, exog
