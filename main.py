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

# Importazioni dai moduli locali
from config import (
    STREAMLIT_CONFIG, DATA_PATH, COLUMN_NAMES, 
    TAB_TITLES, AVAILABLE_MODELS, UI_TEXTS
)
from models import create_model

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(**STREAMLIT_CONFIG)

# --- FUNZIONI UTILITY ---
@st.cache_data
def load_data():
    """Carica il dataset e converte le date."""
    try:
        df = pd.read_csv(DATA_PATH)
        # Conversione colonna data
        df[COLUMN_NAMES['DATE']] = pd.to_datetime(df[COLUMN_NAMES['DATE']])
        df = df.sort_values(by=COLUMN_NAMES['DATE'])
        return df
    except FileNotFoundError:
        st.error(f"❌ File dati non trovato in: {DATA_PATH}. Verifica che 'cicalino_agg.csv' sia nella cartella.")
        return pd.DataFrame()

# --- MAIN APPLICATION ---
def main():
    st.title(UI_TEXTS['main_title'])
    st.caption(UI_TEXTS['main_caption'])

    # 1. Caricamento Dati
    df = load_data()
    if df.empty:
        st.stop() # Ferma l'app se non ci sono dati

    # 2. Sidebar e Filtri
    st.sidebar.header("Filtri Temporali")
    min_date = df[COLUMN_NAMES['DATE']].min()
    max_date = df[COLUMN_NAMES['DATE']].max()
    
    # Widget selezione date
    date_range = st.sidebar.date_input(
        UI_TEXTS['date_input_label'],
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Applicazione filtro
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        mask = (df[COLUMN_NAMES['DATE']].dt.date >= start_d) & (df[COLUMN_NAMES['DATE']].dt.date <= end_d)
        df_filtered = df.loc[mask]
    else:
        df_filtered = df

    # 3. Creazione Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        TAB_TITLES['data'], 
        TAB_TITLES['line'], 
        TAB_TITLES['dist'], 
        TAB_TITLES['models']
    ])

    # --- TAB 1: DATASET ---
    with tab1:
        st.markdown("### 📄 Visualizzazione Dati")
        st.dataframe(df_filtered, use_container_width=True)
        st.info(f"Righe totali nel periodo selezionato: **{len(df_filtered)}**")

    # --- TAB 2: GRAFICI LINEARI ---
    with tab2:
        st.markdown("### 📈 Andamento nel Tempo")
        
        # Grafico Target (Insetti)
        fig = px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['TARGET'], 
                      title="Conteggio Insetti (Target)", template="plotly_white")
        fig.update_traces(line_color='#FF4B4B') # Rosso Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Grafici Meteo (affiancati)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['TEMPERATURE'], 
                                    title="Temperatura (°C)", color_discrete_sequence=['orange']), use_container_width=True)
        with col_b:
            st.plotly_chart(px.line(df_filtered, x=COLUMN_NAMES['DATE'], y=COLUMN_NAMES['HUMIDITY'], 
                                    title="Umidità (%)", color_discrete_sequence=['cyan']), use_container_width=True)

    # --- TAB 3: DISTRIBUZIONI ---
    with tab3:
        st.markdown("### 📊 Distribuzione dei Valori")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df_filtered, x=COLUMN_NAMES['TARGET'], nbins=30, 
                                    title="Distribuzione Conteggi Insetti", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
             fig_box = px.box(df_filtered, y=COLUMN_NAMES['TARGET'], title="Boxplot Insetti")
             st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 4: MODELLI & PREDIZIONI ---
    with tab4:
        st.markdown("### 🤖 Machine Learning & Forecasting")
        st.info("Nota: I modelli Deep Learning (MLP/LSTM) richiedono più tempo per l'addestramento.")
        
        # Selezione Modello
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_model_name = st.selectbox(UI_TEXTS['model_select_label'], AVAILABLE_MODELS)
        
        with col_btn:
            st.write("") # Spaziatura verticale
            st.write("")
            run_btn = st.button(UI_TEXTS['run_button_label'], type="primary", use_container_width=True)

        # Logica di Esecuzione
        if run_btn:
            with st.spinner(f"Addestramento modello **{selected_model_name}** in corso... attendere prego."):
                try:
                    # 1. Creazione istanza modello
                    model = create_model(selected_model_name)
                    
                    # 2. Training e Valutazione
                    # Ritorna (trained_model, metrics_dict)
                    _, metrics = model.train_and_evaluate(df_filtered)
                    
                    st.success(UI_TEXTS['success_message'])
                    
                    # 3. Visualizzazione KPI Metriche
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Train RMSE", f"{metrics['train']['rmse']:.2f}")
                    m2.metric("Test RMSE", f"{metrics['test']['rmse']:.2f}", help="Errore quadratico medio su dati mai visti")
                    m3.metric("Train MAE", f"{metrics['train']['mae']:.2f}")
                    m4.metric("Test MAE", f"{metrics['test']['mae']:.2f}", help="Errore medio assoluto su dati mai visti")
                    
                    st.divider()

                    # 4. Grafico Predizioni vs Realtà (Test Set)
                    st.subheader("Risultati sul Test Set")
                    
                    # Recupero dati per il plot
                    y_test = metrics['y_test']
                    y_pred = metrics['y_test_pred']
                    
                    # Gestione asse X (date o indice numerico)
                    if 'idx_test' in metrics and metrics['idx_test'] is not None:
                        x_axis = metrics['idx_test']
                    else:
                        # CORREZIONE QUI: list() attorno a range()
                        x_axis = list(range(len(y_test)))

                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(x=x_axis, y=y_test, mode='lines', name='Dato Reale (Test)', line=dict(color='gray')))
                    fig_res.add_trace(go.Scatter(x=x_axis, y=y_pred, mode='lines', name='Predizione Modello', line=dict(color='red', dash='dot')))
                    
                    fig_res.update_layout(
                        title=f"Performance: {selected_model_name}", 
                        xaxis_title="Data" if 'idx_test' in metrics else "Step Temporali", 
                        yaxis_title="Numero Insetti",
                        height=450
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

                    # 5. Grafico Loss (solo per Neural Networks)
                    if 'history' in metrics and metrics['history'] is not None:
                        st.subheader("Curve di Apprendimento (Loss)")
                        hist = metrics['history'].history
                        
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(y=hist['loss'], name='Training Loss'))
                        if 'val_loss' in hist:
                            fig_loss.add_trace(go.Scatter(y=hist['val_loss'], name='Validation Loss'))
                        
                        fig_loss.update_layout(title="Andamento Loss durante le epoche", xaxis_title="Epoca", yaxis_title="MSE")
                        st.plotly_chart(fig_loss, use_container_width=True)

                # GESTIONE ERRORI SPECIFICI
                except ImportError as e:
                    if "TensorFlow" in str(e):
                        st.error("❌ Errore: TensorFlow non è disponibile.")
                        st.warning("Assicurati che requirements.txt contenga `tensorflow-cpu`.")
                    else:
                        st.error(f"Errore di importazione: {e}")
                
                except ValueError as e:
                    st.error(f"Errore nei dati o parametri: {e}")
                
                except Exception as e:
                    st.error(f"Si è verificato un errore imprevisto: {e}")
                    with st.expander("Dettagli errore tecnico"):
                        st.exception(e)

if __name__ == "__main__":
    main()
