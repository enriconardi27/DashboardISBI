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
import streamlit as st
from pathlib import Path
from typing import Tuple

from config import COLUMN_NAMES, REQUIRED_COLUMNS

# =======================================================
# CARICAMENTO E VALIDAZIONE DEI DATI
# =======================================================

@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Carica e valida il dataset CSV con dati entomologici e meteorologici.
    
    Questa funzione:
    1. Legge il file CSV specificato
    2. Verifica la presenza delle colonne richieste
    3. Converte la colonna DateTime in formato datetime
    4. Ordina i dati cronologicamente
    
    Args:
        csv_path (Path): Percorso al file CSV da caricare
        
    Returns:
        pd.DataFrame: DataFrame pulito e ordinato cronologicamente
        
    Raises:
        ValueError: Se mancano colonne essenziali nel dataset
    """
    # Caricamento del file CSV
    df = pd.read_csv(csv_path)
    
    # Controllo integrità: verifica presenza di tutte le colonne richieste
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    # Preprocessing temporale: conversione e ordinamento cronologico
    df[COLUMN_NAMES['DATE']] = pd.to_datetime(df[COLUMN_NAMES['DATE']])
    df = df.sort_values(COLUMN_NAMES['DATE']).reset_index(drop=True)
    return df


def filter_data_by_date_range(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    """
    Filtra il dataset in base a un intervallo di date.
    
    Args:
        df (pd.DataFrame): Dataset originale
        date_range (tuple): Tupla con (data_inizio, data_fine)
        
    Returns:
        pd.DataFrame: Dataset filtrato per l'intervallo specificato
    """
    if isinstance(date_range, tuple) and len(date_range) == 2:
        # Applicazione filtro temporale
        filtered_df = df[
            (df[COLUMN_NAMES['DATE']] >= pd.to_datetime(date_range[0])) & 
            (df[COLUMN_NAMES['DATE']] <= pd.to_datetime(date_range[1]))
        ].copy()
        return filtered_df
    else:
        # Fallback: restituisce tutto il dataset se la selezione è incompleta
        return df.copy()


def prepare_dataset_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara il dataset per la visualizzazione in Streamlit.
    Rimuove l'orario dalla colonna DateTime per una visualizzazione più pulita.
    
    Args:
        df (pd.DataFrame): Dataset originale
        
    Returns:
        pd.DataFrame: Dataset preparato per la visualizzazione
    """
    df_view = df.copy()
    df_view[COLUMN_NAMES['DATE']] = df_view[COLUMN_NAMES['DATE']].dt.date
    return df_view


# =======================================================
# PREPROCESSING PER MACHINE LEARNING
# =======================================================

def create_train_test_split(df: pd.DataFrame, test_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crea split cronologico train/test mantenendo l'ordine temporale.
    
    Args:
        df (pd.DataFrame): Dataset completo
        test_split (float): Percentuale per training (default 0.8 = 80%)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    # Calcolo data di split basata sulla percentuale
    min_date = df[COLUMN_NAMES['DATE']].min()
    max_date = df[COLUMN_NAMES['DATE']].max()
    split_date = min_date + (max_date - min_date) * test_split
    
    # Split cronologico
    train_df = df[df[COLUMN_NAMES['DATE']] <= split_date].copy()
    test_df = df[df[COLUMN_NAMES['DATE']] > split_date].copy()
    
    return train_df, test_df, split_date


def create_lagged_features(df: pd.DataFrame, n_lags: int = 5, 
                          target_col: str = None, include_date: bool = True) -> pd.DataFrame:
    """
    Crea features ritardate (lagged) per la modellazione di serie temporali.
    
    Questa funzione:
    1. Prende la serie temporale target
    2. Crea n_lags variabili ritardate (t-1, t-2, ..., t-n_lags)
    3. Include variabili esogene contemporanee
    4. Rimuove righe con valori mancanti
    
    Il lag temporale permette al modello di catturare dipendenze
    sequenziali: il valore al tempo t dipende dai valori t-1, t-2, ecc.
    
    Args:
        df (pd.DataFrame): DataFrame sorgente con serie temporale
        n_lags (int): Numero di ritardi temporali da creare
        target_col (str): Nome colonna variabile target
        include_date (bool): Se includere la colonna data nel risultato
        
    Returns:
        pd.DataFrame: DataFrame con features lagged e righe NaN rimosse
    """
    if target_col is None:
        target_col = COLUMN_NAMES['TARGET']
    
    # Selezione colonne essenziali
    cols_to_use = [COLUMN_NAMES['TARGET'], COLUMN_NAMES['TEMPERATURE'], COLUMN_NAMES['HUMIDITY']]
    if include_date:
        cols_to_use.insert(0, COLUMN_NAMES['DATE'])
    
    work = df[cols_to_use].copy()
    
    # Creazione features ritardate: shift della variabile target
    for i in range(1, n_lags + 1):
        work[f"lag_{i}"] = work[target_col].shift(i)
    
    # Rimozione righe con NaN (create dallo shift) e reset indice
    work = work.dropna().reset_index(drop=True)
    return work


def create_lagged_features_multivariate(df: pd.DataFrame, n_lags: int = 5, 
                                       target_col: str = None, exog_cols: list = None) -> pd.DataFrame:
    """
    Crea features lagged per multiple variabili (target + esogene).
    Necessario per modelli come LSTM che considerano sequenze di tutte le variabili.
    
    Args:
        df (pd.DataFrame): DataFrame sorgente
        n_lags (int): Numero di lag temporali
        target_col (str): Nome colonna target
        exog_cols (list): Lista colonne variabili esogene
        
    Returns:
        pd.DataFrame: DataFrame con features lagged per tutte le variabili
    """
    if target_col is None:
        target_col = COLUMN_NAMES['TARGET']
    if exog_cols is None:
        exog_cols = [COLUMN_NAMES['TEMPERATURE'], COLUMN_NAMES['HUMIDITY']]
    
    lagged = df.copy()
    variables = [target_col] + exog_cols
    
    # Creazione lag per ogni variabile
    for col in variables:
        for i in range(1, n_lags + 1):
            lagged[f"{col}_lag_{i}"] = lagged[col].shift(i)
    
    lagged = lagged.dropna().copy()
    return lagged


# =======================================================
# PREPROCESSING SPECIFICO PER VARMAX
# =======================================================

def prepare_varmax_data(df: pd.DataFrame, rolling_window: int = 7, 
                       ema_alpha: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara i dati specificamente per il modello VARMAX.
    
    Crea:
    1. Variabili endogene: target + rolling sum + EMA
    2. Variabili esogene: temperatura e umidità
    3. Differenziazione per stazionarietà
    
    Args:
        df (pd.DataFrame): Dataset originale
        rolling_window (int): Finestra per rolling sum
        ema_alpha (float): Alpha per Exponential Moving Average
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (endog_diff, exog_aligned)
    """
    # Preparazione DataFrame con indice temporale
    df_indexed = df.set_index(COLUMN_NAMES['DATE'])
    
    # Serie target principale
    y_full = df_indexed[COLUMN_NAMES['TARGET']]
    
    # Feature engineering: variabili derivate dalla serie target
    roll_sum = y_full.rolling(rolling_window, min_periods=1).sum().rename("insetti_roll7")
    ema = y_full.ewm(alpha=ema_alpha, adjust=False).mean().rename("insetti_ema")
    
    # Concatenazione variabili endogene
    endog_raw = pd.concat([y_full.rename("insetti"), roll_sum, ema], axis=1)
    
    # Variabili esogene meteorologiche
    exog_raw = df_indexed[[COLUMN_NAMES['TEMPERATURE'], COLUMN_NAMES['HUMIDITY']]].copy()
    
    # Differenziazione per stazionarietà
    endog_diff = endog_raw.diff().dropna()
    exog_aligned = exog_raw.loc[endog_diff.index].copy()
    
    return endog_diff, exog_aligned


# =======================================================
# UTILITÀ PER PREPROCESSING
# =======================================================

def get_date_range_info(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Restituisce il range temporale del dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: (min_date, max_date)
    """
    min_date = df[COLUMN_NAMES['DATE']].min()
    max_date = df[COLUMN_NAMES['DATE']].max()
    return min_date, max_date


def validate_split_quality(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          min_train: int = 10, min_test: int = 3) -> bool:
    """
    Valida la qualità dello split train/test.
    
    Args:
        train_df (pd.DataFrame): Dataset di training
        test_df (pd.DataFrame): Dataset di test
        min_train (int): Numero minimo di campioni training
        min_test (int): Numero minimo di campioni test
        
    Returns:
        bool: True se lo split è valido, False altrimenti
    """
    return len(train_df) >= min_train and len(test_df) >= min_test