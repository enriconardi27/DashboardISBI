"""plotting_utils.py
============================================================
Modulo di utilità per visualizzazioni interattive con Plotly
e integrazione con Streamlit.

Descrizione
---------
Questo modulo raccoglie funzioni di plotting riutilizzabili per l'analisi
esplorativa e la visualizzazione dei risultati di modelli di machine
learning applicati al dataset "cicalino_agg.csv" incluso nel progetto.
Le funzioni generano figure Plotly (plotly.express / plotly.graph_objects)
e le renderizzano direttamente in un'app Streamlit tramite
`st.plotly_chart` o altri componenti Streamlit.

Dipendenze e costanti
---------------------
Le funzioni qui definite si basano su alcune costanti importate da
`config.py`:
 - COLUMN_NAMES: dizionario con i nomi delle colonne usate (es. 'DATE',
     'TEMPERATURE', 'HUMIDITY', ...)
 - PLOT_CONFIG: dizionario con impostazioni di stile globali (altezza,
     margini, palette di colori, ecc.)
 - SUBPLOT_CONFIG: dimensioni specifiche per figure con subplot

I pacchetti principali utilizzati sono:
 - plotly.express, plotly.graph_objects, plotly.subplots
 - streamlit
 - pandas, numpy

Esempio d'uso (in un'app Streamlit)
-------------------------------
>>> import streamlit as st
>>> from plotting_utils import create_line_plot
>>> df = pd.read_csv('cicalino_agg.csv', parse_dates=['date'])
>>> create_line_plot(df, y_col='insetti', title='Andamento insetti nel tempo')

Note
----
- Le funzioni si aspettano serie o DataFrame con indici temporali
    (pandas.DatetimeIndex) quando rilevante. Se gli indici non sono
    coerenti, convertire prima con `pd.to_datetime` e `set_index`.
- Le funzioni non restituiscono oggetti ma eseguono il rendering in
    Streamlit; per test locale è possibile modificare il codice per
    restituire le figure Plotly invece di chiamare `st.plotly_chart`.

Funzioni disponibili
--------------------
- create_line_plot(df, y_col, title):
    Crea e renderizza un grafico a linee per una serie temporale.

- create_histogram_plot(df, col, title):
    Crea un istogramma con boxplot marginale per visualizzare la
    distribuzione di una variabile.

- create_scatter_temp_humidity(df, size_col, title):
    Scatter plot che mette in relazione temperatura e umidità; la
    dimensione e il colore dei punti rappresentano il conteggio degli
    insetti (o altra colonna numerica passata in `size_col`).

- create_arimax_plot(y_actual, fitted_train, forecast_test, confidence_intervals,
                     model_order, train_metrics, test_metrics):
    Visualizza risultati e intervalli di confidenza di un modello ARIMAX,
    comparando osservati, fitted e forecast.

- create_varmax_plot(endog_actual, fitted_train, forecast_test,
                    confidence_intervals, model_order, train_metrics, test_metrics):
    Visualizza risultati per modelli VARMAX su serie differenziate.

- create_ensemble_model_plot(y_train, y_test, y_train_pred, y_test_pred,
                            model_name, train_metrics, test_metrics):
    Crea un layout 2x2 per confrontare actual vs predicted su train e test
    per modelli ensemble (es. Random Forest, Gradient Boosting).

- create_neural_network_plot(y_train, y_test, y_train_pred, y_test_pred,
                           history, model_name, train_metrics, test_metrics,
                           idx_train=None, idx_test=None):
    Visualizza actual vs fitted, residuals e curve di apprendimento (loss)
    per modelli neurali (MLP, LSTM). Accetta opzionali indici temporali.

- display_metrics(train_metrics, test_metrics):
    Mostra le metriche principali (RMSE, MAE) in due colonne con componenti
    Streamlit `st.metric`.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np

from config import COLUMN_NAMES, PLOT_CONFIG, SUBPLOT_CONFIG

# =======================================================
# GRAFICI BASE PER ANALISI ESPLORATIVA
# =======================================================

def create_line_plot(df: pd.DataFrame, y_col: str, title: str) -> None:
    """
    Crea un grafico a linee interattivo per visualizzare l'andamento temporale di una variabile.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati da visualizzare
        y_col (str): Nome della colonna da rappresentare sull'asse Y
        title (str): Titolo del grafico
    """
    # Creazione grafico a linee con Plotly Express
    fig = px.line(df, x=COLUMN_NAMES['DATE'], y=y_col, title=title)
    
    # Personalizzazione layout: dimensioni e margini ottimizzati per la dashboard
    fig.update_layout(
        height=PLOT_CONFIG['height'], 
        margin=PLOT_CONFIG['margin']
    )
    
    # Rendering del grafico in Streamlit con larghezza adattiva
    st.plotly_chart(fig, width='stretch')


def create_histogram_plot(df: pd.DataFrame, col: str, title: str) -> None:
    """
    Crea un istogramma con boxplot marginale per analizzare la distribuzione di una variabile.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati
        col (str): Nome della colonna di cui analizzare la distribuzione
        title (str): Titolo del grafico
    """
    # Istogramma con 30 bin e boxplot marginale per analisi distribuzione completa
    fig = px.histogram(
        df, 
        x=col, 
        nbins=PLOT_CONFIG['histogram_bins'], 
        marginal='box', 
        title=title
    )
    
    # Layout ottimizzato per la visualizzazione in dashboard
    fig.update_layout(
        height=PLOT_CONFIG['height'], 
        margin=PLOT_CONFIG['margin']
    )
    
    # Rendering con larghezza adattiva
    st.plotly_chart(fig, width='stretch')


def create_scatter_temp_humidity(df: pd.DataFrame, size_col: str, title: str) -> None:
    """
    Crea uno scatter plot interattivo per analizzare la relazione temperatura-umidità.
    
    Il grafico utilizza:
    - Posizione: temperatura (X) vs umidità (Y)
    - Dimensione punti: proporzionale al numero di insetti
    - Colore: mappa il numero di insetti con scala di colori Viridis
    - Hover: informazioni dettagliate al passaggio del mouse
    
    Args:
        df (pd.DataFrame): DataFrame con i dati meteorologici e entomologici
        size_col (str): Nome della colonna che determina la dimensione dei punti
        title (str): Titolo del grafico
    """
    # Scatter plot con mappatura dimensione e colore basata sui conteggi di insetti
    fig = px.scatter(
        df,
        x=COLUMN_NAMES['TEMPERATURE'],        # Asse X: temperatura
        y=COLUMN_NAMES['HUMIDITY'],           # Asse Y: umidità
        size=size_col,                        # Dimensione punti: numero insetti
        color=size_col,                       # Colore punti: numero insetti
        color_continuous_scale=PLOT_CONFIG['color_scale'],  # Scala colori scientifica
        size_max=PLOT_CONFIG['scatter_size_max'],           # Dimensione massima punti
        hover_data={                          # Dati mostrati nell'hover tooltip
            COLUMN_NAMES['DATE']: True, 
            size_col: True, 
            COLUMN_NAMES['TEMPERATURE']: True, 
            COLUMN_NAMES['HUMIDITY']: True
        },
        title=title,
    )
    
    # Personalizzazione estetica: trasparenza per sovrapposizioni
    fig.update_traces(marker=dict(opacity=PLOT_CONFIG['scatter_opacity']))
    
    # Layout e colorbar personalizzata
    fig.update_layout(
        height=PLOT_CONFIG['height'], 
        margin=PLOT_CONFIG['margin'], 
        coloraxis_colorbar=dict(title='Numero insetti')  # Titolo legenda colori
    )
    
    # Rendering con larghezza che si adatta al container
    st.plotly_chart(fig, width='stretch')


# =======================================================
# GRAFICI PER MODELLI MACHINE LEARNING
# =======================================================

def create_arimax_plot(y_actual: pd.Series, fitted_train: pd.Series, 
                      forecast_test: pd.Series, confidence_intervals: pd.DataFrame,
                      model_order: tuple, train_metrics: dict, test_metrics: dict) -> None:
    """
    Crea visualizzazione per risultati modello ARIMAX.
    
    Args:
        y_actual (pd.Series): Serie temporale reale completa
        fitted_train (pd.Series): Valori fittati sul training
        forecast_test (pd.Series): Forecast sul test set
        confidence_intervals (pd.DataFrame): Intervalli di confidenza
        model_order (tuple): Ordine del modello ARIMAX (p,d,q)
        train_metrics (dict): Metriche training {'rmse': float, 'mae': float}
        test_metrics (dict): Metriche test {'rmse': float, 'mae': float}
    """
    fig = go.Figure()

    # Serie temporale reale (valori osservati)
    fig.add_trace(go.Scatter(
        x=y_actual.index,
        y=y_actual.values,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Valori fittati sul training set
    if fitted_train is not None:
        fig.add_trace(go.Scatter(
            x=fitted_train.index,
            y=fitted_train.values,
            mode='lines',
            name='Fitted (Train)',
            line=dict(color='red', dash='dash')
        ))

    # Predizioni sul test set
    if forecast_test is not None:
        fig.add_trace(go.Scatter(
            x=forecast_test.index,
            y=forecast_test.values,
            mode='lines',
            name='Forecast (Test)',
            line=dict(color='green')
        ))

    # Intervalli di confidenza per le predizioni
    if confidence_intervals is not None:
        try:
            lower = confidence_intervals.iloc[:, 0]
            upper = confidence_intervals.iloc[:, 1]
            
            fig.add_trace(go.Scatter(
                x=list(confidence_intervals.index) + list(confidence_intervals.index[::-1]),
                y=list(lower.values) + list(upper.values[::-1]),
                fill='toself',
                fillcolor='rgba(0,200,0,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='95% CI'
            ))
        except Exception:
            pass

    # Configurazione layout del grafico
    fig.update_layout(
        title=(
            f"ARIMAX {model_order} - Numero di insetti (Cicalino)<br>"
            f"Train RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f} | "
            f"Test RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}"
        ),
        xaxis_title='Date',
        yaxis_title='Numero di insetti',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig, width='stretch')


def create_varmax_plot(endog_actual: pd.Series, fitted_train: pd.Series,
                      forecast_test: pd.Series, confidence_intervals: tuple,
                      model_order: tuple, train_metrics: dict, test_metrics: dict) -> None:
    """
    Crea visualizzazione per risultati modello VARMAX.
    
    Args:
        endog_actual (pd.Series): Serie differenziata reale
        fitted_train (pd.Series): Valori fittati training
        forecast_test (pd.Series): Forecast test
        confidence_intervals (tuple): (lower, upper) intervalli confidenza
        model_order (tuple): Ordine modello VARMAX
        train_metrics (dict): Metriche training
        test_metrics (dict): Metriche test
    """
    fig = go.Figure()

    # Serie reale (differenziata)
    fig.add_trace(go.Scatter(
        x=endog_actual.index,
        y=endog_actual.values,
        mode='lines',
        name='Actual (diff)',
        line=dict(color='blue')
    ))

    # Fitted (train)
    fig.add_trace(go.Scatter(
        x=fitted_train.index,
        y=fitted_train.values,
        mode='lines',
        name='Fitted (train, diff)',
        line=dict(color='red', dash='dash')
    ))

    # Forecast (test)
    fig.add_trace(go.Scatter(
        x=forecast_test.index,
        y=forecast_test.values,
        mode='lines',
        name='Forecast (test, diff)',
        line=dict(color='green')
    ))

    # Intervalli di confidenza
    if confidence_intervals and len(confidence_intervals) == 2:
        lower, upper = confidence_intervals
        try:
            fig.add_trace(go.Scatter(
                x=list(lower.index) + list(lower.index[::-1]),
                y=list(lower.values) + list(upper.values[::-1]),
                fill='toself',
                fillcolor='rgba(0,200,0,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='95% CI'
            ))
        except Exception:
            pass

    fig.update_layout(
        title=(
            f"VARMAX — insetti (diff) | order={model_order}<br>"
            f"Train RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f} | "
            f"Test RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}"
        ),
        xaxis_title='Date',
        yaxis_title='Differenced insetti',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig, width='stretch')


def create_ensemble_model_plot(y_train: pd.Series, y_test: pd.Series,
                              y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                              model_name: str, train_metrics: dict, test_metrics: dict) -> None:
    """
    Crea visualizzazione 2x2 per modelli ensemble (Random Forest, Gradient Boosting).
    
    Args:
        y_train (pd.Series): Target training
        y_test (pd.Series): Target test
        y_train_pred (np.ndarray): Predizioni training
        y_test_pred (np.ndarray): Predizioni test
        model_name (str): Nome del modello
        train_metrics (dict): Metriche training
        test_metrics (dict): Metriche test
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Train — Actual vs Fitted (RMSE: {train_metrics['rmse']:.2f})",
        f"Test — Actual vs Forecast (RMSE: {test_metrics['rmse']:.2f})",
        "",
        "",
    ))

    # Train plot
    fig.add_trace(go.Scatter(
        x=y_train.index, 
        y=y_train.values, 
        mode='lines+markers', 
        name='Actual (train)', 
        marker=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=y_train.index, 
        y=y_train_pred, 
        mode='lines+markers', 
        name='Fitted (train)', 
        marker=dict(color='red')
    ), row=1, col=1)

    # Test plot
    fig.add_trace(go.Scatter(
        x=y_test.index, 
        y=y_test.values, 
        mode='lines+markers', 
        name='Actual (test)', 
        marker=dict(color='blue')
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=y_test.index, 
        y=y_test_pred, 
        mode='lines+markers', 
        name='Forecast (test)', 
        marker=dict(color='red')
    ), row=1, col=2)

    fig.update_layout(
        height=SUBPLOT_CONFIG['height'], 
        width=SUBPLOT_CONFIG['width'], 
        showlegend=True, 
        title_text=f"{model_name} — Train/Test"
    )

    st.plotly_chart(fig, width='stretch')


def create_neural_network_plot(y_train: pd.Series, y_test: pd.Series,
                               y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                               history: object, model_name: str, 
                               train_metrics: dict, test_metrics: dict,
                               idx_train: pd.Series = None, idx_test: pd.Series = None) -> None:
    """
    Crea visualizzazione 2x2 per reti neurali (MLP, LSTM) con curve di apprendimento.
    
    Args:
        y_train, y_test: Target series
        y_train_pred, y_test_pred: Predizioni
        history: Oggetto history di Keras con loss curves
        model_name: Nome del modello
        train_metrics, test_metrics: Dizionari con metriche
        idx_train, idx_test: Indici temporali per asse X (opzionali)
    """
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f"Actual vs Fitted — Train\nRMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}",
        "Residuals — Train",
        "Model Loss over Epochs (MSE)",
        f"Forecast vs Actual — Test\nRMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}",
    ))

    # Usa indici temporali se forniti, altrimenti indici numerici
    x_train = idx_train if idx_train is not None else (y_train.index if hasattr(y_train, 'index') else range(len(y_train)))
    x_test = idx_test if idx_test is not None else (y_test.index if hasattr(y_test, 'index') else range(len(y_test)))

    # Gestione dati: converte a array se necessario (per compatibilità pandas/numpy)
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test

    # Plot 1: Train actual vs fitted
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train_values, 
        mode='lines+markers', name='Actual (train)', 
        marker=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train_pred, 
        mode='lines+markers', name='Fitted (train)', 
        marker=dict(color='green')
    ), row=1, col=1)

    # Plot 2: Residuals
    residuals = y_train_values - y_train_pred
    fig.add_trace(go.Scatter(
        x=x_train, y=residuals, 
        mode='lines+markers', name='Residuals', 
        marker=dict(color='purple')
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[x_train.min(), x_train.max()], y=[0,0], 
        mode='lines', name='Zero', 
        line=dict(color='black', dash='dash')
    ), row=1, col=2)

    # Plot 3: Learning curves
    epochs = list(range(1, len(history.history['loss'])+1))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history['loss'], 
        mode='lines', name='Training Loss', 
        marker=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history.history['val_loss'], 
        mode='lines', name='Validation Loss', 
        marker=dict(color='red')
    ), row=2, col=1)

    # Plot 4: Test forecast vs actual
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test_values, 
        mode='lines+markers', name='Actual (test)', 
        marker=dict(color='blue')
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test_pred, 
        mode='lines+markers', name='Forecast (test)', 
        marker=dict(color='green')
    ), row=2, col=2)

    # Layout specifico per LSTM con altezza maggiore
    height = SUBPLOT_CONFIG['lstm_height'] if 'LSTM' in model_name else SUBPLOT_CONFIG['height']
    
    fig.update_layout(
        height=height, 
        width=SUBPLOT_CONFIG['width'], 
        showlegend=True, 
        title_text=f'{model_name} — Train/Test'
    )
    
    st.plotly_chart(fig, width='stretch')


# =======================================================
# UTILITÀ DI SUPPORTO
# =======================================================

def display_metrics(train_metrics: dict, test_metrics: dict) -> None:
    """
    Visualizza le metriche di performance in formato pulito.
    
    Args:
        train_metrics (dict): Metriche training con chiavi 'rmse', 'mae'
        test_metrics (dict): Metriche test con chiavi 'rmse', 'mae'
    """
    st.success("Valutazione completata")
    
    # Layout a due colonne per organizzazione visiva delle metriche
    c1, c2 = st.columns(2)
    
    # Colonna 1: Metriche Training Set
    with c1:
        st.metric("Train RMSE", f"{train_metrics['rmse']:.3f}")
        st.metric("Train MAE", f"{train_metrics['mae']:.3f}")
    
    # Colonna 2: Metriche Test Set
    with c2:
        st.metric("Test RMSE", f"{test_metrics['rmse']:.3f}")
        st.metric("Test MAE", f"{test_metrics['mae']:.3f}")