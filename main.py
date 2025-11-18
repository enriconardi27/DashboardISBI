# main.py
# =====================================================================
# DASHBOARD PRINCIPALE - ORCHESTRATORE MODULI
#
# Scopo:
#   File entry-point per l'app Streamlit. Coordina il caricamento dei dati,
#   l'inizializzazione dell'interfaccia utente, la navigazione a tab e
#   l'esecuzione dei diversi modelli di forecasting/analisi.
#
# Contesto operativo:
#   - Progettato per essere eseguito con `streamlit run main.py`.
#   - Usa Streamlit per UI, pandas/numpy per manipolazione dati e moduli
#     locali per plotting e modelli (models.py, plotting_utils.py, data_utils.py).
#   - La configurazione (percorso dati, nomi colonne, parametri modello) è
#     centralizzata in `config.py`.
#
# Principali responsabilità:
#   - Inizializzare l'interfaccia utente tramite `ui_components.initialize_ui()`.
#   - Caricare e preprocessare il dataset (funzione `load_data` in `data_utils.py`).
#   - Fornire i controlli interattivi per filtrare il dataset.
#   - Creare la navigazione a tab e popolare ogni tab con contenuti specifici.
#   - Dispatch delle chiamate ai vari esecutori modello (ARIMAX, VARMAX,
#     ensemble, reti neurali) e visualizzare i risultati + metriche.
#   - Gestione degli errori a livello applicazione (FileNotFoundError, ecc.).
#
# Formato dati atteso:
#   - Un CSV (o DataFrame) con una colonna temporale e le colonne specificate in
#     `config.COLUMN_NAMES` (es. 'insetti' come target, 'temp_med', 'umid_med', ...).
#   - Il dataset viene caricato da `DATA_PATH` (definito in `config.py`).
#
# Dipendenze principali:
#   - streamlit as st
#   - pandas as pd
#   - numpy as np
#   - moduli locali: config, data_utils, ui_components, plotting_utils, models
#
# Note operative importanti:
#   - `st.set_page_config(...)` (chiamata da `ui_components.setup_streamlit_page()`)
#     deve essere invocata prima di renderizzare altri elementi Streamlit per
#     evitare warning/errore di configurazione.
#   - Alcune funzioni di plotting e di preprocessing sono importate localmente
#     nelle loro funzioni per ridurre il tempo di import iniziale.
#   - Alcune routine (es. uso di TensorFlow) richiedono controlli di disponibilità:
#     la funzione `execute_neural_network_model` verifica `TF_AVAILABLE` prima
#     di procedere.
#
# Esportato / Funzioni principali DEFINITE IN QUESTO FILE:
#   - execute_arimax_model(df_raw: pd.DataFrame) -> None
#       Prepara y (target) e X (esogeni), applica split cronologico, esegue grid
#       search per ARIMAX, produce grafici e metriche, gestisce eccezioni locali.
#
#   - execute_varmax_model(df_raw: pd.DataFrame) -> None
#       Prepara e allinea variabili endogene/exogene, applica il modello VARMAX,
#       calcola predizioni, intervalli di confidenza, mostra grafici e metriche.
#
#   - execute_ensemble_model(df_raw: pd.DataFrame, model_name: str) -> None
#       Wrapper per eseguire modelli ensemble (Random Forest, Gradient Boosting):
#       costruisce, allena, valuta e visualizza risultati + metriche.
#
#   - execute_neural_network_model(df_raw: pd.DataFrame, model_name: str) -> None
#       Controlla availability di TensorFlow, costruisce e allena la rete (MLP/LSTM),
#       visualizza curve di apprendimento, predizioni e metriche. Chiama
#       `show_error_message('tensorflow', model_name)` + `st.stop()` se TF non disponibile.
#
#   - execute_selected_model(model_choice: str, df_raw: pd.DataFrame) -> None
#       Dispatcher che mappa la stringa scelta dall'utente alla funzione di
#       esecuzione corretta. Gestisce modello non implementato con errore utente.
#
#   - main() -> None
#       Flusso principale:
#         1. initialize_ui()
#         2. load_data(DATA_PATH)
#         3. handle_data_filtering(df_raw)   # widget di filtro date
#         4. create_tab_navigation()         # ottiene tab
#         5. render_*_tab(...) per ogni tab
#         6. render_models_tab_header(df_raw) e, se richiesto, esecuzione modello
#       Gestisce eccezioni di alto livello ed errori di I/O.
#
# ENTRY POINT:
#   - Lo script è eseguibile come modulo principale: `if __name__ == "__main__": main()`
#
# Suggerimenti per sviluppo e debugging:
#   - Per test rapidi, eseguire `streamlit run main.py` dalla root del progetto.
#   - Per verificare configurazione Streamlit, controllare `STREAMLIT_CONFIG` in `config.py`.
#   - Se vuoi mostrare la sidebar informativa definita in `ui_components.render_sidebar_info()`,
#     chiamala esplicitamente dopo `initialize_ui()` (non viene chiamata automaticamente).
#
# Error handling e robustezza:
#   - main() cattura FileNotFoundError e Exception generica e notifica l'utente con `st.error`.
#   - Le funzioni modello hanno blocchi try/except locali per evitare crash dell'intera app.
#
# =====================================================================


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Importazioni locali
from config import (
    STREAMLIT_CONFIG, DATA_PATH, COLUMN_NAMES, 
    TAB_TITLES, AVAILABLE_MODELS, UI_TEXTS, PLOT_CONFIG
)
from models import create_model

# Configurazione Pagina
st.set_page_config(**STREAMLIT_CONFIG)

# --- FUNZIONI DI CARICAMENTO ---
@st.cache_data
def load_data():
    try:
        # Caricamento dataset
        df = pd.read_csv(DATA_PATH)
        
        # Conversione data
        df[COLUMN_NAMES['DATE']] = pd.to_datetime(df[COLUMN_NAMES['DATE']])
        df = df.sort_values(by=COLUMN_NAMES['DATE'])
        
        return df
    except FileNotFoundError:
        st.error(f"File dati non trovato in: {DATA_PATH}. Assicurati di aver caricato 'cicalino_agg.csv'.")
        return pd.DataFrame()

# --- MAIN APP ---
def main():
    st.title(UI_TEXTS['main_title'])
    st.caption(UI_TEXTS['main_caption'])

    # Caricamento Dati
    df = load_data()
    if df.empty:
        return

    # Sidebar - Filtri
    st.sidebar.header("Filtri")
    min_date = df[COLUMN_NAMES['DATE']].min()
    max_date = df[COLUMN_NAMES['DATE']].max()
    
    date_range = st.sidebar.date_input(
        UI_TEXTS['date_input_label'],
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filtro dati in base alla data
    if len(date_range) == 2:
        mask = (df[COLUMN_NAMES['DATE']].dt.date >= date_range[0]) & (df[COLUMN_NAMES['DATE']].dt.date <= date_range[1])
        df_filtered = df.loc[mask]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        TAB_TITLES['data'], 
        TAB_TITLES['line'], 
        TAB_TITLES['dist'], 
        TAB_TITLES['models']
    ])

    # TAB 1: Dati
    with tab1:
        st.dataframe(df_filtered, use_container_width=True)
        st.info(f"Righe totali: {len(df_filtered)}")

    # TAB 2: Line Plot
    with tab2:
        st.subheader("Andamento Temporale")
        fig = px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['TARGET'], 
                      title="Target nel tempo", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Meteo
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['TEMPERATURE'], 
                                    title="Temperatura", color_discrete_sequence=['orange']), use_container_width=True)
        with col_b:
            st.plotly_chart(px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['HUMIDITY'], 
                                    title="Umidità", color_discrete_sequence=['cyan']), use_container_width=True)

    # TAB 3: Distribuzioni
    with tab3:
        st.subheader("Distribuzione Variabili")
        fig_hist = px.histogram(df_filtered, x=COLUMN_NAMES['TARGET'], nbins=30, title="Istogramma Target")
        st.plotly_chart(fig_hist, use_container_width=True)

    # TAB 4: Modelli
    with tab4:
        st.subheader("Forecasting & Analisi")
        
        col_model, col_btn = st.columns([3, 1])
        with col_model:
            selected_model = st.selectbox(UI_TEXTS['model_select_label'], AVAILABLE_MODELS)
        
        with col_btn:
            st.write("") # Spacer
            st.write("") 
            run_btn = st.button(UI_TEXTS['run_button_label'], type="primary")

        if run_btn:
            with st.spinner(f"Addestramento {selected_model} in corso..."):
                try:
                    # Istanzia il modello
                    model_instance = create_model(selected_model)
                    
                    # Training e valutazione
                    # Nota: train_and_evaluate restituisce (model, metrics_dict)
                    _, metrics = model_instance.train_and_evaluate(df_filtered)
                    
                    st.success(UI_TEXTS['success_message'])
                    
                    # Visualizzazione Metriche
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Train RMSE", f"{metrics['train']['rmse']:.2f}")
                    m_col2.metric("Test RMSE", f"{metrics['test']['rmse']:.2f}")
                    m_col3.metric("Train MAE", f"{metrics['train']['mae']:.2f}")
                    m_col4.metric("Test MAE", f"{metrics['test']['mae']:.2f}")
                    
                    # Plot Predizioni vs Reale (Test Set)
                    st.subheader("Risultati sul Test Set")
                    
                    # Recuperiamo i dati per il plot
                    y_test = metrics['y_test']
                    y_pred = metrics['y_test_pred']
                    
                    # Se ci sono indici temporali nel dizionario metrics (MLP/LSTM li mettono), usiamoli
                    if 'idx_test' in metrics:
                        x_axis = metrics['idx_test']
                    else:
                        # Fallback generico se l'indice non è passato esplicitamente
                        x_axis = range(len(y_test))

                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(x=x_axis, y=y_test, mode='lines', name='Reale'))
                    fig_res.add_trace(go.Scatter(x=x_axis, y=y_pred, mode='lines', name='Predetto', line=dict(dash='dot')))
                    
                    fig_res.update_layout(title=f"Predizione {selected_model}", height=400)
                    st.plotly_chart(fig_res, use_container_width=True)

                    # Se c'è la history (NN), mostriamola
                    if 'history' in metrics and metrics['history'] is not None:
                        st.subheader("Loss durante il training")
                        loss = metrics['history'].history['loss']
                        val_loss = metrics['history'].history['val_loss']
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=loss, name='Train Loss'))
                        fig_loss.add_trace(go.Scatter(y=val_loss, name='Val Loss'))
                        st.plotly_chart(fig_loss, use_container_width=True)

                except ImportError as e:
                    st.error(f"Errore libreria mancante: {e}")
                    if "TensorFlow" in str(e):
                        st.warning("Suggerimento: Assicurati che tensorflow-cpu sia nel requirements.txt")
                except Exception as e:
                    st.error(f"Errore durante l'esecuzione: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()