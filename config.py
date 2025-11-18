# config.py
# =====================================================================
# File di configurazione centrale per la Dashboard Insetti Cicalino
# Contiene tutte le costanti, parametri e configurazioni utilizzate nell'applicazione
# =====================================================================

from pathlib import Path

# =======================================================
# CONFIGURAZIONE STREAMLIT
# =======================================================
STREAMLIT_CONFIG = {
    "page_title": "Dashboard Insetti — Cicalino",
    "page_icon": "🐞", 
    "layout": "wide"
}

# =======================================================
# CONFIGURAZIONE DATI E PERCORSI
# =======================================================

# Percorso al dataset principale
DATA_PATH = Path(__file__).parent / 'cicalino_agg.csv'

# Nomi delle colonne del dataset (alias per leggibilità)
COLUMN_NAMES = {
    'DATE': 'DateTime',
    'TARGET': 'Numero di insetti_Cicalino',
    'TEMPERATURE': 'Media Temperatura_Cicalino', 
    'HUMIDITY': 'Media Umidità_Cicalino'
}

# Colonne richieste nel dataset
REQUIRED_COLUMNS = [
    'DateTime',
    'Numero di insetti_Cicalino',
    'Media Temperatura_Cicalino',
    'Media Umidità_Cicalino'
]

# =======================================================
# CONFIGURAZIONE MODELLI MACHINE LEARNING
# =======================================================

# Configurazione generale
MODEL_CONFIG = {
    'test_split': 0.8,  # 80% training, 20% testing
    'random_state': 42  # Seed per riproducibilità
}

# Parametri per modelli con features lagged
LAGGED_FEATURES_CONFIG = {
    'n_lags': 5,  # Numero di lag temporali
    'exog_cols': [COLUMN_NAMES['TEMPERATURE'], COLUMN_NAMES['HUMIDITY']]
}

# Ordini ARIMA da testare nella grid search
ARIMAX_ORDERS = [(0,0,0), (1,0,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1)]

# Parametri ottimali per Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None
}

# Parametri ottimali per Gradient Boosting  
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 3
}

# Configurazione MLP (Multi-Layer Perceptron)
MLP_CONFIG = {
    'architecture': [32, 16, 8],  # Hidden layers
    'learning_rate': 1e-3,
    'epochs': 500,
    'batch_size': 16,
    'callbacks': {
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 20,
            'min_lr': 1e-5
        },
        'early_stopping': {
            'monitor': 'val_loss', 
            'patience': 40,
            'restore_best_weights': True
        }
    }
}

# Configurazione LSTM
LSTM_CONFIG = {
    'lstm_units': 50,
    'dense_units': 32,
    'dropout_rate': 0.1,
    'learning_rate': 1e-3,
    'epochs': 300,
    'batch_size': 8,
    'callbacks': {
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5, 
            'patience': 10,
            'min_lr': 1e-5
        },
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 20,
            'restore_best_weights': True
        }
    }
}

# =======================================================
# CONFIGURAZIONE VISUALIZZAZIONI
# =======================================================

# Configurazione grafici Plotly
PLOT_CONFIG = {
    'height': 350,
    'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20},
    'color_scale': 'Viridis',
    'histogram_bins': 30,
    'scatter_size_max': 30,
    'scatter_opacity': 0.7
}

# Configurazione subplot per modelli ML
SUBPLOT_CONFIG = {
    'height': 700,
    'width': 1000,
    'lstm_height': 800
}

# =======================================================
# CONFIGURAZIONE UI
# =======================================================

# Configurazione filtro temporale 
DATE_FILTER_CONFIG = {
    'default_days': 90,  # Ultimi 90 giorni come default
    'format': "YYYY-MM-DD"
}

# Titoli e nomi dei tab
TAB_TITLES = {
    'data': "📄 Dataset",
    'line': "📈 Line plot", 
    'dist': "📊 Distribuzioni",
    'models': "🤖 Modelli & Forecasting"
}

# Lista modelli disponibili
AVAILABLE_MODELS = [
    "ARIMAX",
    "VARMAX", 
    "Random Forest",
    "Gradient Boosting",
    "MLP (lagged)",
    "LSTM (lagged)"
]

# =======================================================
# MESSAGGI E TESTI DELL'INTERFACCIA
# =======================================================

UI_TEXTS = {
    'main_title': "🐞 Dashboard — Cicalino: insetti & meteo",
    'main_caption': "Dataset incorporato: unione stazioni Cicalino. Visualizzazione, distribuzioni e modelli di forecasting.",
    'period_label': "**Periodo disponibile:**",
    'date_input_label': "Seleziona intervallo per visualizzazioni",
    'model_select_label': "Scegli il modello",
    'run_button_label': "Esegui modello",
    'success_message': "Valutazione completata",
    'split_info': "Data di split train/test (fissa 80%): {} — tutto **≤** va in Train; il resto in Test",
    'warning_split': "Intervallo di split troppo sbilanciato. Aumenta il train o il test.",
    'error_tensorflow': "TensorFlow non è disponibile in questo ambiente: impossibile eseguire {}."
}

# =======================================================
# CONFIGURAZIONE FEATURE ENGINEERING
# =======================================================

# Parametri per VARMAX
VARMAX_CONFIG = {
    'order': (1, 0),  # Ordine ottimale (p,q)
    'rolling_window': 7,  # Finestra rolling per somma mobile
    'ema_alpha': 0.3  # Alpha per Exponential Moving Average
}