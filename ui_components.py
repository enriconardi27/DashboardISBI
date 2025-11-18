# ui_components.py
# =====================================================================
# Modulo: ui_components
# Scopo:
#   Contiene funzioni riutilizzabili per costruire e gestire l'interfaccia
#   utente Streamlit dell'applicazione. L'obiettivo è separare la logica
#   di presentazione (layout, widget, messaggi, tab) dal resto dell'app,
#   offrendo API semplici per inizializzare la pagina e renderizzare i
#   diversi contenuti (dataset, grafici, distribuzioni, modelli).
#
# Principali responsabilità:
#   - Configurazione della pagina Streamlit (titolo, layout, ecc.)
#   - Rendering di controlli interattivi (filtri per data, selectbox, pulsanti)
#   - Creazione e popolamento dei tab dell'app (dataset, grafici, distribuzioni, modelli)
#   - Utility per messaggi informativi, avvisi e presentazione delle metriche
#   - Funzioni di orchestrazione per inizializzare e filtrare i dati
#
# Dipendenze esterne (import principali):
#   - streamlit as st
#   - pandas as pd
#   - datetime.timedelta
#   - config (STREAMLIT_CONFIG, COLUMN_NAMES, DATE_FILTER_CONFIG, TAB_TITLES, AVAILABLE_MODELS, UI_TEXTS)
#   - data_utils (get_date_range_info, prepare_dataset_for_display, filter_data_by_date_range,
#                 create_train_test_split, validate_split_quality)
#   - plotting_utils (importato localmente nelle funzioni che generano grafici)
#
# Formato dati atteso:
#   - Un DataFrame Pandas contenente dati temporali (colonna data/tempo o indice di data)
#   - Le colonne usate dall'app sono definite in `config.COLUMN_NAMES` (es. 'TEMPERATURE', 'HUMIDITY', 'TARGET')
#
# Esportato (funzioni pubbliche principali):
#   - setup_streamlit_page(): Configura le impostazioni di pagina Streamlit.
#   - render_main_header(): Mostra titolo e sottotitolo principali.
#   - render_date_filter(df): Renderizza il controllo per selezionare l'intervallo temporale e
#       restituisce il range selezionato (start_date, end_date).
#   - render_model_selector(): Selectbox per scegliere il modello ML disponibile.
#   - render_run_button(): Pulsante per eseguire il modello.
#   - render_dataset_tab(df): Visualizza il dataset preprocessato in una tabella interattiva.
#   - render_line_plots_tab(df): Visualizza grafici temporali e scatter per analisi esplorativa.
#   - render_distributions_tab(df): Visualizza istogrammi delle variabili principali.
#   - render_models_tab_header(df_raw): Mostra informazioni e controlli per la sezione modelli
#       (calcolo split, scelta modello, pulsante di esecuzione) e ritorna (split_date, model_choice, run_flag).
#   - create_tab_navigation(): Crea e restituisce gli oggetti tab Streamlit usati per la navigazione.
#   - show_success_message(), show_error_message(...), show_info_message(), show_warning_message():
#       Helper per visualizzare feedback all'utente.
#   - create_metrics_columns(), render_metric(): Utility per presentare metriche (RMSE, MAE, ecc.).
#   - add_visual_separator(), render_subheader(): Piccole utility di layout.
#   - create_responsive_columns(num_cols, ratios=None): Crea colonne responsive per layout adattivo.
#   - create_expander(title, expanded=False): Crea un expander Streamlit per contenuto opzionale.
#   - render_sidebar_info(): Popola la sidebar con informazioni e descrizione dei modelli/metriche.
#   - initialize_ui(): Inizializza la UI chiamando setup e header principale.
#   - handle_data_filtering(df_raw): Coordina il widget di filtro data e applica il filtro ai dati grezzi.
#
# Note di implementazione:
#   - Per ridurre il tempo di avvio alcuni moduli pesanti (es. `plotting_utils`) sono importati
#     all'interno delle funzioni che li usano.
#   - Tutti i testi, etichette e configurazioni sono centralizzati in `config.py` per facilitare
#     manutenzione e localizzazione.
#
# =====================================================================

import streamlit as st
import pandas as pd
from datetime import timedelta
from typing import Tuple, Optional

from config import (
    STREAMLIT_CONFIG, COLUMN_NAMES, DATE_FILTER_CONFIG, 
    TAB_TITLES, AVAILABLE_MODELS, UI_TEXTS
)
from data_utils import get_date_range_info, prepare_dataset_for_display

# Note IMPORTANTE:
# - `streamlit` è il framework usato per creare l'interfaccia web interattiva.
# - `pandas` è utilizzato per la manipolazione dei DataFrame che rappresentano i dati.
# - Alcune costanti e testi sono centralizzati in `config.py` per facilitare
#   la localizzazione e la manutenzione (etichette, nomi colonne, opzioni ecc.).
# - `data_utils` contiene funzioni di utilità per il preprocessing e per ottenere
#   informazioni sul range di date del dataset.

# =======================================================
# CONFIGURAZIONE INIZIALE STREAMLIT
# =======================================================

def setup_streamlit_page() -> None:
    """Configura le impostazioni base della pagina Streamlit."""
    # Impostiamo la configurazione della pagina usando i parametri definiti in config
    # Questo include titolo della pagina, icona, layout, e altre preferenze di Streamlit.
    st.set_page_config(**STREAMLIT_CONFIG)


def render_main_header() -> None:
    """Renderizza l'intestazione principale della dashboard."""
    # Visualizziamo il titolo principale e una caption sotto di esso.
    # I testi sono presi da UI_TEXTS per centralizzare la gestione dei contenuti.
    st.title(UI_TEXTS['main_title'])
    st.caption(UI_TEXTS['main_caption'])


# =======================================================
# CONTROLLI INTERATTIVI
# =======================================================

def render_date_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, tuple]:
    """
    Renderizza il filtro per la selezione dell'intervallo temporale.
    
    Args:
        df (pd.DataFrame): Dataset completo
        
    Returns:
        tuple: (dataset_filtrato, range_date_selezionato)
    """
    # Calcolo del range temporale disponibile
    min_date, max_date = get_date_range_info(df)
    # get_date_range_info ritorna la data minima e massima presenti nel DataFrame,
    # tipicamente usate per impostare i limiti del widget di selezione delle date.
    
    # Layout a due colonne per informazioni e controlli
    col_a, col_b = st.columns(2)
    
    # Colonna A: Informazioni periodo disponibile
    with col_a:
        # Mostriamo all'utente l'intervallo temporale complessivo disponibile nel dataset
        st.write(f"{UI_TEXTS['period_label']} {min_date.date()} → {max_date.date()}")
    
    # Colonna B: Selettore data
    with col_b:
        # Default: ultimi N giorni per performance migliori
        default_start = max(
            min_date, 
            max_date - pd.Timedelta(days=DATE_FILTER_CONFIG['default_days'])
        )
        # Il valore di default per l'inizio del range è calcolato come 'max_date - default_days',
        # ma non viene fatto scendere al di sotto della data minima presente nel dataset.
        
        # Widget selezione intervallo date
        date_range = st.date_input(
            UI_TEXTS['date_input_label'],
            value=(default_start.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            format=DATE_FILTER_CONFIG['format']
        )
    
    return date_range


def render_model_selector() -> str:
    """
    Renderizza il selettore per la scelta del modello ML.
    
    Returns:
        str: Nome del modello selezionato
    """
    # Ritorna la stringa corrispondente al modello scelto dall'utente
    return st.selectbox(UI_TEXTS['model_select_label'], AVAILABLE_MODELS)


def render_run_button() -> bool:
    """
    Renderizza il pulsante per eseguire il modello.
    
    Returns:
        bool: True se il pulsante è stato premuto
    """
    # Quando il bottone viene premuto Streamlit ritorna True per il singolo ciclo di esecuzione
    # (utilizzato per avviare il training/forecasting quando l'utente lo richiede)
    return st.button(UI_TEXTS['run_button_label'])


# =======================================================
# CONTENUTI DEI TAB
# =======================================================

def render_dataset_tab(df: pd.DataFrame) -> None:
    """
    Renderizza il contenuto del tab Dataset.
    
    Args:
        df (pd.DataFrame): Dataset da visualizzare
    """
    # Sottotitolo della sezione
    st.subheader("Vista dataset")
    
    # Preprocessing per visualizzazione pulita: prepare_dataset_for_display applica
    # tipicamente rinomina colonne, formatta date e rimuove campi non necessari.
    df_display = prepare_dataset_for_display(df)
    
    # Rendering della tabella: st.dataframe crea una tabella interattiva con supporto
    # a ordinamento e scrolling. Impostiamo dimensioni per coerenza visiva.
    st.dataframe(
        df_display, 
        width='stretch',  # Larghezza adattiva
        height=420        # Altezza fissa per consistenza UI
    )


def render_line_plots_tab(df: pd.DataFrame) -> None:
    """
    Renderizza il contenuto del tab Line Plots.
    
    Args:
        df (pd.DataFrame): Dataset per i grafici
    """
    # Import locale per evitare di caricare moduli pesanti all'avvio dell'app
    from plotting_utils import create_line_plot, create_scatter_temp_humidity
    
    # Sottotitolo della sezione grafici
    st.subheader("Line plot variabili principali")
    
    # Usiamo un layout a tre colonne per mostrare tre grafici affiancati
    c1, c2, c3 = st.columns(3)
    
    # Grafico 1: temperatura media nel tempo
    with c1:
        create_line_plot(df, COLUMN_NAMES['TEMPERATURE'], "Media Temperatura Cicalino")
    
    # Grafico 2: numero di insetti (target) nel tempo
    with c2:
        create_line_plot(df, COLUMN_NAMES['TARGET'], "Numero di insetti Cicalino")
    
    # Grafico 3: umidità media nel tempo
    with c3:
        create_line_plot(df, COLUMN_NAMES['HUMIDITY'], "Media Umidità Cicalino")

    # Separatore visivo tra gruppi di grafici
    st.markdown("---")

    # Sezione per analizzare la relazione tra temperatura e umidità
    st.subheader("Relazione temperatura vs umidità")
    create_scatter_temp_humidity(
        df, 
        COLUMN_NAMES['TARGET'], 
        "Relazione tra temperatura, umidità e catture (Cicalino)"
    )


def render_distributions_tab(df: pd.DataFrame) -> None:
    """
    Renderizza il contenuto del tab Distribuzioni.
    
    Args:
        df (pd.DataFrame): Dataset per l'analisi distributiva
    """
    # Import locale per funzioni di plotting legate agli istogrammi
    from plotting_utils import create_histogram_plot
    
    st.subheader("Distribuzioni")
    
    # Layout a tre colonne per comparare le distribuzioni delle tre variabili principali
    c1, c2, c3 = st.columns(3)
    
    # Istogramma del target (numero di insetti)
    with c1:
        create_histogram_plot(df, COLUMN_NAMES['TARGET'], "Distribuzione — Numero di insetti")
    
    # Istogramma della temperatura media
    with c2:
        create_histogram_plot(df, COLUMN_NAMES['TEMPERATURE'], "Distribuzione — Media Temperatura")
    
    # Istogramma dell'umidità media
    with c3:
        create_histogram_plot(df, COLUMN_NAMES['HUMIDITY'], "Distribuzione — Media Umidità")


def render_models_tab_header(df_raw: pd.DataFrame) -> Tuple[pd.Timestamp, str, bool]:
    """
    Renderizza l'intestazione del tab Modelli con informazioni e controlli.
    
    Args:
        df_raw (pd.DataFrame): Dataset completo
        
    Returns:
        tuple: (split_date, model_choice, run_button_pressed)
    """
    # Import locale di utilità per ottenere range di date e configurazione del modello
    from data_utils import get_date_range_info
    from config import MODEL_CONFIG
    
    # Sottotitolo per la sezione modelli/forecasting
    st.subheader("Modelli e forecasting (train/test nel periodo completo)")

    # Calcoliamo la data di split in base alla percentuale definita in MODEL_CONFIG
    # split_date è la soglia temporale che separa train e test.
    min_date, max_date = get_date_range_info(df_raw)
    split_date = min_date + (max_date - min_date) * MODEL_CONFIG['test_split']
    
    # Mostriamo all'utente informazioni su come è stato calcolato lo split
    st.info(UI_TEXTS['split_info'].format(split_date.date()))
    
    # Render del selettore di modello: l'utente sceglie quale modello eseguire
    model_choice = render_model_selector()
    
    # Validazione della qualità dello split: usiamo funzioni di data_utils per creare
    # i dataset train/test e valutare se lo split è sensato (es. non vuoto o sbilanciato)
    from data_utils import create_train_test_split, validate_split_quality
    train_df, test_df, _ = create_train_test_split(df_raw, MODEL_CONFIG['test_split'])
    
    # Se la validazione fallisce, informiamo l'utente con un warning
    if not validate_split_quality(train_df, test_df):
        st.warning(UI_TEXTS['warning_split'])
    
    # Mostriamo il pulsante che avvia l'esecuzione (training/prediction)
    run_button = render_run_button()
    
    # Ritorniamo la data di split, la scelta del modello e il flag che indica
    # se l'utente ha premuto il pulsante di esecuzione
    return split_date, model_choice, run_button


# =======================================================
# GESTIONE TAB NAVIGATION
# =======================================================

def create_tab_navigation() -> Tuple:
    """
    Crea la struttura di navigazione a tab.
    
    Returns:
        tuple: Oggetti tab per il contenuto
    """
    # Crea le tab dell'app con titoli presi da config.TAB_TITLES
    return st.tabs([
        TAB_TITLES['data'],
        TAB_TITLES['line'], 
        TAB_TITLES['dist'],
        TAB_TITLES['models']
    ])


# =======================================================
# MESSAGGI E NOTIFICHE
# =======================================================

def show_success_message() -> None:
    """Mostra messaggio di successo per completamento operazione."""
    # Messaggio verde di conferma (usato dopo operazioni correttamente completate)
    st.success(UI_TEXTS['success_message'])


def show_error_message(error_type: str, model_name: str = "") -> None:
    """
    Mostra messaggi di errore appropriati.
    
    Args:
        error_type (str): Tipo di errore ('tensorflow', 'data', 'model')
        model_name (str): Nome del modello (per errori specifici)
    """
    # Selezione del testo di errore in base al tipo fornito
    if error_type == 'tensorflow':
        # Messaggio specifico per errori legati a TensorFlow (es. import o esecuzione)
        st.error(UI_TEXTS['error_tensorflow'].format(model_name))
    elif error_type == 'data':
        # Errore nel caricamento o processamento dei dati
        st.error("Errore nel caricamento o processamento dei dati.")
    elif error_type == 'model':
        # Errore generico relativo all'esecuzione del modello selezionato
        st.error(f"Errore nell'esecuzione del modello {model_name}.")
    else:
        # Messaggio di errore fallback
        st.error("Errore generico nell'applicazione.")


def show_info_message(message: str) -> None:
    """Mostra messaggio informativo."""
    # Messaggio blu informativo
    st.info(message)


def show_warning_message(message: str) -> None:
    """Mostra messaggio di warning."""
    # Messaggio di avvertimento (giallo)
    st.warning(message)


# =======================================================
# UTILITÀ UI
# =======================================================

def create_metrics_columns() -> Tuple:
    """
    Crea layout a due colonne per visualizzazione metriche.
    
    Returns:
        tuple: (colonna_sinistra, colonna_destra)
    """
    # Restituisce due colonne affiancate per visualizzare metriche (es. RMSE e MAE)
    return st.columns(2)


def render_metric(label: str, value: float, precision: int = 3) -> None:
    """
    Renderizza una singola metrica con formattazione.
    
    Args:
        label (str): Etichetta della metrica
        value (float): Valore della metrica
        precision (int): Precisione decimale
    """
    # Formatta il valore numerico con la precisione desiderata e lo mostra come metrica
    st.metric(label, f"{value:.{precision}f}")


def add_visual_separator() -> None:
    """Aggiunge un separatore visivo."""
    # Separatore orizzontale
    st.markdown("---")


def render_subheader(text: str) -> None:
    """Renderizza un sottotitolo."""
    # Piccolo wrapper che rende più leggibile il codice chiamante
    st.subheader(text)


# =======================================================
# LAYOUT RESPONSIVE
# =======================================================

def create_responsive_columns(num_cols: int, ratios: list = None) -> Tuple:
    """
    Crea colonne responsive per layout adattivo.
    
    Args:
        num_cols (int): Numero di colonne
        ratios (list): Rapporti di larghezza delle colonne (opzionale)
        
    Returns:
        tuple: Oggetti colonna
    """
    # Se vengono forniti i ratios usiamo quelli per dimensionare le colonne,
    # altrimenti creiamo un numero uniforme di colonne
    if ratios:
        return st.columns(ratios)
    else:
        return st.columns(num_cols)


def create_expander(title: str, expanded: bool = False):
    """
    Crea un contenitore espandibile per contenuto opzionale.
    
    Args:
        title (str): Titolo dell'expander
        expanded (bool): Se inizialmente espanso
        
    Returns:
        Oggetto expander Streamlit
    """
    # Ritorna l'oggetto expander che può essere usato come context manager
    return st.expander(title, expanded=expanded)


# =======================================================
# SIDEBAR (OPZIONALE)
# =======================================================

def render_sidebar_info() -> None:
    """Renderizza informazioni nella sidebar (se necessario)."""
    # Popolamento della sidebar con informazioni utili per l'utente
    with st.sidebar:
        # Intestazione e descrizione generale
        st.header("ℹ️ Informazioni")
        st.write("Dashboard per l'analisi di dati entomologici e meteorologici della stazione Cicalino.")
        
        # Lista dei modelli disponibili (presa dalla configurazione)
        st.header("📊 Modelli Disponibili")
        for model in AVAILABLE_MODELS:
            st.write(f"• {model}")
        
        # Spiegazione breve delle metriche esposte
        st.header("📈 Metriche")
        st.write("• **RMSE**: Root Mean Square Error")
        st.write("• **MAE**: Mean Absolute Error")
        
        # Breve riepilogo della configurazione predefinita mostrata all'utente
        st.header("🔧 Configurazione")
        st.write("• Split Train/Test: 80/20")
        st.write("• Ordine cronologico preservato")
        st.write("• Post-processing: clipping + rounding")


# =======================================================
# FUNZIONI DI ORCHESTRAZIONE
# =======================================================

def initialize_ui() -> None:
    """Inizializza tutti i componenti base dell'UI."""
    # Chiamate iniziali per impostare la pagina e mostrare l'intestazione principale
    setup_streamlit_page()
    render_main_header()


def handle_data_filtering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Gestisce il workflow completo di filtro dei dati.
    
    Args:
        df_raw (pd.DataFrame): Dataset originale
        
    Returns:
        pd.DataFrame: Dataset filtrato
    """
    from data_utils import filter_data_by_date_range
    # Mostriamo il widget per la selezione dell'intervallo di date e applichiamo
    # il filtro sul DataFrame originale. La funzione filter_data_by_date_range
    # si occupa di gestire sia tuple di date che casi limite.
    date_range = render_date_filter(df_raw)
    return filter_data_by_date_range(df_raw, date_range)