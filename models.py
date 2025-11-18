# models.py
# =====================================================================
# Modulo: implementazioni e wrapper per i modelli usati nell'app
#
# Scopo
# Questo modulo centralizza le implementazioni dei modelli utilizzati nella
# applicazione Streamlit. Fornisce wrapper per modelli statistici (ARIMAX,
# VARMAX), una classe base per modelli ensemble (Random Forest, Gradient
# Boosting), implementazioni per reti neurali (MLP e LSTM) e utility per il
# calcolo delle metriche. Infine espone una factory `create_model(model_name)`
# per istanziare i modelli a partire da una stringa (usata dalla UI).
#
# Contenuti principali
# - ARIMAXModel: grid-search manuale sugli ordini ARIMA (scelta via AIC)
# - VARMAXModel: wrapper per VARMAX multivariato con pre-processing
# - EnsembleModel: classe base che implementa pipeline lagging → split →
#   scaling → fit → predict → metriche
#   - RandomForestModel e GradientBoostingModel: semplici wrapper che
#     passano la classe sklearn corrispondente e i parametri predefiniti
# - MLPModel, LSTMModel: implementazioni Keras/TensorFlow per dati laggati
#
# Contratto e convenzioni
# - Tutti i metodi `train_and_evaluate` (o simili) ritornano una tupla
#   (modello_addestrato, metrics_dict). Il `metrics_dict` contiene almeno
#   i campi 'train' e 'test' con metriche (RMSE, MAE) e le predizioni
#   corrispondenti ('y_train_pred', 'y_test_pred').
# - Le feature laggati sono generate tramite le funzioni in `data_utils.py`
#   (ad es. `create_lagged_features`, `create_lagged_features_multivariate`).
# - Le configurazioni (nomi colonne, parametri modelli, numero di lag, ecc.)
#   vengono lette da `config.py`.
# - TensorFlow è opzionale: se non è installato, le classi MLP/LSTM solleveranno
#   ImportError al momento dell'istanza. La UI dovrebbe evitare di proporre
#   modelli non disponibili.
#
# Note pratiche
# - ARIMAX utilizza una grid search manuale sugli ordini definiti in
#   `ARIMAX_ORDERS` e seleziona il modello con AIC più basso.
# - Ensemble e NN usano parametri predefiniti da `config.py`; non è previsto
#   tunning automatico a meno che non venga aggiunto esplicitamente (es. via
#   GridSearchCV o Keras Tuner).
# - Le predizioni vengono spesso post-processate con `np.clip(..., 0, None)` e
#   `.round()` perché l'app lavora con conteggi (valori non negativi interi).
#
# Esempio rapido d'uso
#   from models import create_model
#   m = create_model('Random Forest')
#   trained_model, metrics = m.train_and_evaluate(df)
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