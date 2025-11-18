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

# data_utils.py
# =====================================================================
# Modulo: Funzioni di preparazione e manipolazione dati
# =====================================================================

import pandas as pd
import numpy as np
from config import COLUMN_NAMES

def create_lagged_features(df: pd.DataFrame, n_lags: int, include_date: bool = False) -> pd.DataFrame:
    """
    Crea feature ritardate (lag) solo per la colonna target.
    Usato da: Random Forest, Gradient Boosting, MLP, ARIMAX.
    
    Args:
        df (pd.DataFrame): Dataset originale.
        n_lags (int): Numero di passi temporali indietro da considerare.
        include_date (bool): Se mantenere la colonna data nel dataframe risultante.
        
    Returns:
        pd.DataFrame: Dataset con le colonne lag_1, lag_2, ..., lag_n.
    """
    df_lagged = df.copy()
    target = COLUMN_NAMES['TARGET']
    
    # Creazione colonne lag
    for t in range(1, n_lags + 1):
        df_lagged[f'lag_{t}'] = df_lagged[target].shift(t)
    
    # Rimuove le righe iniziali che contengono NaN a causa dello shift
    df_lagged = df_lagged.dropna()
    
    # Gestione colonna data (se presente e richiesta)
    if not include_date and COLUMN_NAMES['DATE'] in df_lagged.columns:
        # La data viene mantenuta se serve per i plot, altrimenti ignorata dalle feature X
        pass
        
    return df_lagged


def create_lagged_features_multivariate(df: pd.DataFrame, n_lags: int, target_col: str, exog_cols: list) -> pd.DataFrame:
    """
    Crea feature ritardate per target E variabili esogene.
    Usato da: LSTM (che richiede la storia completa di tutte le feature).
    
    Args:
        df (pd.DataFrame): Dataset.
        n_lags (int): Numero di lag.
        target_col (str): Nome colonna target.
        exog_cols (list): Lista nomi colonne esogene.
        
    Returns:
        pd.DataFrame: Dataset con lag per tutte le variabili specificate.
    """
    df_copy = df.copy()
    cols_to_lag = [target_col] + exog_cols
    
    # Crea lag per ogni colonna specificata (target + meteo)
    for col in cols_to_lag:
        for t in range(1, n_lags + 1):
            df_copy[f'{col}_lag_{t}'] = df_copy[col].shift(t)
            
    return df_copy.dropna()


def prepare_varmax_data(df: pd.DataFrame, rolling_window: int, ema_alpha: float):
    """
    Prepara i dati specificamente per il modello VARMAX.
    Esegue smoothing e differenziazione per rendere la serie stazionaria.
    
    CORREZIONE IMPORTANTE:
    Include 'Temperatura' come variabile ENDOGENA insieme al Target.
    Questo risolve l'errore "Only gave one variable to VAR".
    
    Args:
        df (pd.DataFrame): Dataset completo.
        rolling_window (int): (Non usato in questa versione con EMA, mantenuto per compatibilità).
        ema_alpha (float): Fattore di smoothing esponenziale.
        
    Returns:
        tuple: (endog_data, exog_data) pronte per il fitting.
    """
    # Selezione colonne dai config
    target = COLUMN_NAMES['TARGET']
    temp = COLUMN_NAMES['TEMPERATURE']
    humid = COLUMN_NAMES['HUMIDITY']
    
    # Copia del dataset limitata alle colonne di interesse
    data = df[[target, temp, humid]].copy()
    
    # 1. Smoothing: riduce il rumore degli insetti (spikes improvvisi)
    # Usiamo Exponential Moving Average
    data_smooth = data.ewm(alpha=ema_alpha).mean()
    
    # 2. Differenziazione: rimuove il trend e la stagionalità di base (Stazionarietà)
    data_diff = data_smooth.diff().dropna()
    
    # 3. Definizione variabili Endogene ed Esogene
    # VARMAX richiede un vettore di variabili endogene (>= 2 variabili per VAR).
    # Modelliamo Insetti e Temperatura come sistema accoppiato (Endogene).
    # L'Umidità rimane come variabile esterna (Esogena).
    
    endog = data_diff[[target, temp]]  # <-- Ora contiene 2 colonne: Insetti e Temp
    exog = data_diff[[humid]]          # <-- Umidità
    
    return endog, exog

