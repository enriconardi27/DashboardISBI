# models.py
# =====================================================================
# Modulo: implementazioni e wrapper per i modelli usati nell'app
# =====================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional, Union

# Librerie per modelli statistici e machine learning
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow (Importazione condizionale per evitare crash su Streamlit Cloud)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Importazioni dai moduli locali
from config import (
    COLUMN_NAMES, ARIMAX_ORDERS, RANDOM_FOREST_PARAMS, GRADIENT_BOOSTING_PARAMS,
    MLP_CONFIG, LSTM_CONFIG, VARMAX_CONFIG, LAGGED_FEATURES_CONFIG
)
from data_utils import (
    create_lagged_features, create_lagged_features_multivariate, 
    prepare_varmax_data
)

# =======================================================
# FUNZIONI UTILITY PER METRICHE
# =======================================================

def calculate_rmse(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mae(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def create_metrics_dict(y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred)
    }

# =======================================================
# MODELLI STATISTICI PER SERIE TEMPORALI
# =======================================================

class ARIMAXModel:
    def __init__(self, orders: list = None):
        self.orders = orders if orders is not None else ARIMAX_ORDERS
        self.best_model = None
        self.best_order = None
        self.best_aic = np.inf
    
    def fit_with_gridsearch(self, y_train: pd.Series, X_train: pd.DataFrame) -> Tuple[object, tuple, float]:
        best_aic = np.inf
        best_order = None
        best_model = None

        for order in self.orders:
            try:
                model_tmp = sm.tsa.ARIMA(endog=y_train, exog=X_train, order=order).fit()
                if model_tmp.aic < best_aic:
                    best_aic = model_tmp.aic
                    best_order = order
                    best_model = model_tmp
            except Exception:
                continue

        self.best_model = best_model
        self.best_order = best_order
        self.best_aic = best_aic

        return best_model, best_order, best_aic

    def train_and_evaluate(self, df: pd.DataFrame):
        # Adattamento interfaccia per coerenza con altri modelli
        # Creiamo lag solo per avere le esogene allineate, ma ARIMAX usa la storia interna
        df_prep = create_lagged_features(df, n_lags=1, include_date=True) # min lag per allineamento
        
        target_col = COLUMN_NAMES['TARGET']
        exog_cols = LAGGED_FEATURES_CONFIG['exog_cols']
        
        # Split 80/20
        split_idx = int(len(df_prep) * 0.8)
        
        y = df_prep[target_col]
        X = df_prep[exog_cols]
        
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        # Grid Search e Fit
        model, _, _ = self.fit_with_gridsearch(y_train, X_train)
        
        if model is None:
            raise ValueError("ARIMAX non è riuscito a convergere con nessuno degli ordini forniti.")

        # Predict
        start = len(y_train)
        end = len(y_train) + len(y_test) - 1
        y_test_pred = model.predict(start=start, end=end, exog=X_test).clip(0, None).round()
        y_train_pred = model.predict(start=0, end=len(y_train)-1, exog=X_train).clip(0, None).round()
        
        metrics = {
            'train': create_metrics_dict(y_train, y_train_pred),
            'test': create_metrics_dict(y_test, y_test_pred),
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'idx_test': df_prep[COLUMN_NAMES['DATE']].iloc[split_idx:]
        }
        return model, metrics


class VARMAXModel:
    def __init__(self, order: tuple = None, rolling_window: int = None, ema_alpha: float = None):
        self.order = order if order is not None else VARMAX_CONFIG['order']
        self.rolling_window = rolling_window if rolling_window is not None else VARMAX_CONFIG['rolling_window']
        self.ema_alpha = ema_alpha if ema_alpha is not None else VARMAX_CONFIG['ema_alpha']
        self.model = None
    
    def prepare_and_fit(self, df: pd.DataFrame) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
        endog_diff, exog_aligned = prepare_varmax_data(
            df, self.rolling_window, self.ema_alpha
        )
        
        split_idx = int(len(endog_diff) * 0.8)
        endog_tr = endog_diff.iloc[:split_idx]
        exog_tr = exog_aligned.iloc[:split_idx]
        
        m = sm.tsa.VARMAX(
            endog_tr,
            exog=exog_tr,
            order=self.order,
            trend='n'
        )
        self.model = m.fit(disp=False)
        return self.model, endog_diff, exog_aligned

    def train_and_evaluate(self, df: pd.DataFrame):
        model, endog_diff, exog_aligned = self.prepare_and_fit(df)
        
        split_idx = int(len(endog_diff) * 0.8)
        
        # Previsione sul test set
        # Nota: VARMAX su dati differenziati richiede integrazione per tornare alla scala originale.
        # Qui per semplicità valutiamo l'errore sui dati trasformati o si dovrebbe reintegrare.
        # Per questa demo, restituiamo le metriche sui dati trasformati (trend/diff).
        
        exog_test = exog_aligned.iloc[split_idx:]
        endog_test = endog_diff.iloc[split_idx:]
        endog_train = endog_diff.iloc[:split_idx]
        
        start = len(endog_train)
        end = len(endog_train) + len(endog_test) - 1
        
        pred = model.predict(start=start, end=end, exog=exog_test)
        target_col = COLUMN_NAMES['TARGET']
        
        # Estrazione colonna target dalle previsioni multivariate
        y_test_pred = pred[target_col].clip(0, None)
        y_test_true = endog_test[target_col]
        
        # Predizione train (approssimata)
        y_train_pred = model.fittedvalues[target_col].iloc[:split_idx].clip(0, None)
        y_train_true = endog_train[target_col]

        metrics = {
            'train': create_metrics_dict(y_train_true, y_train_pred),
            'test': create_metrics_dict(y_test_true, y_test_pred),
            'y_train': y_train_true,
            'y_test': y_test_true,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        return model, metrics

# =======================================================
# MODELLI ENSEMBLE (RANDOM FOREST, GRADIENT BOOSTING)
# =======================================================

class EnsembleModel:
    def __init__(self, model_class, model_params: dict, n_lags: int = None):
        self.model_class = model_class
        self.model_params = model_params
        self.n_lags = n_lags if n_lags is not None else LAGGED_FEATURES_CONFIG['n_lags']
        self.exog_cols = LAGGED_FEATURES_CONFIG['exog_cols']
        self.model = None
        self.scaler = StandardScaler()
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[object, Dict[str, float]]:
        df_lagged = create_lagged_features(df, n_lags=self.n_lags, include_date=True)

        feature_cols = [f"lag_{i}" for i in range(1, self.n_lags + 1)] + self.exog_cols
        
        # Check colonne
        feature_cols = [c for c in feature_cols if c in df_lagged.columns]

        X = df_lagged[feature_cols]
        y = df_lagged[COLUMN_NAMES['TARGET']]
        dates = df_lagged[COLUMN_NAMES['DATE']]

        split_idx = int(len(df_lagged) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = self.model_class(**self.model_params, random_state=42)
        self.model.fit(X_train_s, y_train)

        y_train_pred = np.clip(self.model.predict(X_train_s), 0, None).round()
        y_test_pred = np.clip(self.model.predict(X_test_s), 0, None).round()

        metrics = {
            'train': create_metrics_dict(y_train, y_train_pred),
            'test': create_metrics_dict(y_test, y_test_pred),
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'idx_test': dates.iloc[split_idx:]
        }
        return self.model, metrics


class RandomForestModel(EnsembleModel):
    def __init__(self):
        super().__init__(RandomForestRegressor, RANDOM_FOREST_PARAMS)


class GradientBoostingModel(EnsembleModel):
    def __init__(self):
        super().__init__(GradientBoostingRegressor, GRADIENT_BOOSTING_PARAMS)


# =======================================================
# MODELLI NEURAL NETWORK
# =======================================================

class MLPModel:
    def __init__(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow non disponibile. Verifica requirements.txt.")
        
        self.n_lags = LAGGED_FEATURES_CONFIG['n_lags']
        self.exog_cols = LAGGED_FEATURES_CONFIG['exog_cols']
        self.config = MLP_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build_model(self, input_shape: int):
        inputs = tf.keras.Input(shape=(input_shape,))
        x = inputs
        for units in self.config['architecture']:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss="mse"
        )
        return model
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        tf.random.set_seed(42)
        np.random.seed(42)

        df_lagged = create_lagged_features(df, n_lags=self.n_lags, include_date=True)
        feature_cols = [f"lag_{i}" for i in range(1, self.n_lags + 1)] + self.exog_cols
        feature_cols = [c for c in feature_cols if c in df_lagged.columns]

        X = df_lagged[feature_cols].astype(float)
        y = df_lagged[COLUMN_NAMES['TARGET']].astype(float)
        
        split_idx = int(len(df_lagged) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = self.build_model(X_train_s.shape[1])

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(**self.config['callbacks']['reduce_lr'], verbose=0),
            tf.keras.callbacks.EarlyStopping(**self.config['callbacks']['early_stopping'], verbose=0),
        ]

        self.history = self.model.fit(
            X_train_s, y_train.values,
            validation_data=(X_test_s, y_test.values),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=0,
            callbacks=callbacks
        )

        y_train_pred = np.clip(self.model.predict(X_train_s, verbose=0).flatten(), 0, None).round()
        y_test_pred = np.clip(self.model.predict(X_test_s, verbose=0).flatten(), 0, None).round()

        metrics = {
            'train': create_metrics_dict(y_train, y_train_pred),
            'test': create_metrics_dict(y_test, y_test_pred),
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'history': self.history,
            'idx_test': df_lagged[COLUMN_NAMES['DATE']].iloc[split_idx:]
        }
        return self.model, metrics


class LSTMModel:
    def __init__(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow non disponibile. Verifica requirements.txt.")
        
        self.n_lags = LAGGED_FEATURES_CONFIG['n_lags']
        self.exog_cols = LAGGED_FEATURES_CONFIG['exog_cols']
        self.config = LSTM_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build_model(self, n_timesteps: int, n_features: int):
        inputs = tf.keras.Input(shape=(n_timesteps, n_features))
        x = tf.keras.layers.LSTM(self.config['lstm_units'], activation='tanh')(inputs)
        x = tf.keras.layers.Dense(self.config['dense_units'], activation='relu')(x)
        x = tf.keras.layers.Dropout(self.config['dropout_rate'])(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config['learning_rate']),
            loss='mse'
        )
        return model
    
    def prepare_3d_data(self, df: pd.DataFrame):
        lagged_df = create_lagged_features_multivariate(
            df, n_lags=self.n_lags, 
            target_col=COLUMN_NAMES['TARGET'], 
            exog_cols=self.exog_cols
        )
        variables = [COLUMN_NAMES['TARGET']] + self.exog_cols

        timestep_cols = []
        for t in range(self.n_lags):
            lag = self.n_lags - t
            cols = [f"{var}_lag_{lag}" for var in variables]
            timestep_cols.append(cols)

        X_list = [lagged_df[cols].values for cols in timestep_cols]
        X = np.stack(X_list, axis=1)
        y = lagged_df[COLUMN_NAMES['TARGET']].values
        
        # Gestione date se presente (per plot)
        dates = None
        if COLUMN_NAMES['DATE'] in df.columns:
             # Dobbiamo allineare le date tagliando le prime n_lags righe
             dates = df[COLUMN_NAMES['DATE']].iloc[self.n_lags:]
             # Assicuriamoci che la lunghezza corrisponda dopo dropna nel lagging
             if len(dates) > len(y):
                 dates = dates.iloc[-len(y):]

        return X, y, dates
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[object, Dict]:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        X, y, dates = self.prepare_3d_data(df)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        idx_test = dates.iloc[split_idx:] if dates is not None else None

        n_samples_tr, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        X_train_scaled_2d = self.scaler.fit_transform(X_train_2d)
        X_test_scaled_2d = self.scaler.transform(X_test_2d)

        X_train_s = X_train_scaled_2d.reshape(n_samples_tr, n_timesteps, n_features).astype(np.float32)
        X_test_s = X_test_scaled_2d.reshape(X_test.shape[0], n_timesteps, n_features).astype(np.float32)
        
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        self.model = self.build_model(n_timesteps, n_features)
        
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(**self.config['callbacks']['reduce_lr'], verbose=0),
            tf.keras.callbacks.EarlyStopping(**self.config['callbacks']['early_stopping'], verbose=0),
        ]

        self.history = self.model.fit(
            X_train_s, y_train,
            validation_data=(X_test_s, y_test),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=0,
            callbacks=callbacks
        )

        y_train_pred = np.clip(self.model.predict(X_train_s, verbose=0).flatten(), 0, None).round()
        y_test_pred = np.clip(self.model.predict(X_test_s, verbose=0).flatten(), 0, None).round()

        metrics = {
            'train': create_metrics_dict(y_train, y_train_pred),
            'test': create_metrics_dict(y_test, y_test_pred),
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'history': self.history,
            'idx_test': idx_test
        }
        return self.model, metrics


# =======================================================
# FACTORY PATTERN PER CREAZIONE MODELLI
# =======================================================

def create_model(model_name: str):
    """
    Factory function per creare istanze dei modelli.
    """
    model_mapping = {
        "ARIMAX": ARIMAXModel,
        "VARMAX": VARMAXModel, 
        "Random Forest": RandomForestModel,
        "Gradient Boosting": GradientBoostingModel,
        "MLP (lagged)": MLPModel,
        "LSTM (lagged)": LSTMModel
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Modello {model_name} non supportato")
    
    return model_mapping[model_name]()
