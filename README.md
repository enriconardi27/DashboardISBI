# 🐞 Dashboard Insetti Cicalino - Versione Modulare

## 📋 Panoramica

Dashboard interattiva per l'analisi e predizione di dati entomologici della stazione Cicalino. 

## 🏗️ Architettura

### 📁 Struttura File

```
streamlit_PG/
├── main.py                    # 🎯 File principale orchestratore
├── config.py                  # ⚙️  Configurazioni e costanti
├── data_utils.py             # 📊 Gestione e preprocessing dati
├── plotting_utils.py         # 📈 Funzioni di visualizzazione
├── models.py                 # 🤖 Implementazioni modelli ML
├── ui_components.py          # 🖥️  Componenti interfaccia utente
├── cicalino_agg.csv         # 📋 Dataset principale
├── requirements.txt         # 📦 Dipendenze Python
└── README.md               # 📖 Documentazione
```

### 🔧 Moduli e Responsabilità

#### `config.py`
- **Scopo**: Centralizza tutte le configurazioni
- **Contiene**:
  - Parametri Streamlit
  - Configurazioni modelli ML
  - Costanti UI e plotting
  - Nomi colonne dataset

#### `data_utils.py`
- **Scopo**: Gestione completa dei dati
- **Funzioni principali**:
  - Caricamento e validazione CSV
  - Preprocessing e feature engineering
  - Split train/test cronologico
  - Preparazione dati per VARMAX e LSTM

#### `plotting_utils.py`
- **Scopo**: Tutte le visualizzazioni interattive
- **Grafici disponibili**:
  - Line plots temporali
  - Istogrammi con boxplot
  - Scatter plots correlazioni
  - Visualizzazioni specifiche per ogni modello ML

#### `models.py`
- **Scopo**: Implementazioni modelli machine learning
- **Modelli inclusi**:
  - ARIMAX (con grid search)
  - VARMAX (multivariato)
  - Random Forest & Gradient Boosting
  - MLP e LSTM (deep learning)

#### `ui_components.py`
- **Scopo**: Componenti interfaccia utente
- **Funzionalità**:
  - Setup pagina Streamlit
  - Controlli interattivi
  - Gestione tab navigation
  - Messaggi e notifiche

#### `main.py`
- **Scopo**: Orchestrazione generale
- **Responsabilità**:
  - Coordinamento tra moduli
  - Workflow principale applicazione
  - Gestione errori centralizzata
  - Entry point esecuzione

## 🚀 Come Eseguire

### 1. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Esecuzione Dashboard
```bash
streamlit run main.py
```

## 📦 Dipendenze Principali

- **streamlit**: Framework web interattivo
- **pandas**: Manipolazione dati
- **numpy**: Operazioni numeriche
- **plotly**: Visualizzazioni interattive
- **scikit-learn**: Modelli ML tradizionali
- **statsmodels**: Modelli statistici (ARIMA, VARMAX)
- **tensorflow**: Reti neurali

## 🔄 Workflow di Utilizzo

1. **Caricamento**: `data_utils` carica e valida il dataset
2. **Filtering**: UI permette selezione periodo temporale
3. **Visualizzazione**: Tab per esplorazione dati
4. **Modelling**: Selezione ed esecuzione modelli ML
5. **Risultati**: Grafici e metriche automatiche

## 📊 Modelli Disponibili

### Statistici
- **ARIMAX**: Serie temporali con variabili esogene
- **VARMAX**: Modello vettoriale multivariato

### Machine Learning
- **Random Forest**: Ensemble di alberi decisionali
- **Gradient Boosting**: Boosting sequenziale

### Deep Learning
- **MLP**: Multi-Layer Perceptron con features lagged
- **LSTM**: Long Short-Term Memory per serie temporali

## 🎨 Interfaccia Utente

### Tab Principali
1. **📄 Dataset**: Visualizzazione tabellare dati
2. **📈 Line Plot**: Grafici temporali variabili
3. **📊 Distribuzioni**: Analisi statistiche
4. **🤖 Modelli**: Training e forecasting

### Controlli Interattivi
- Filtro periodo temporale
- Selezione modello ML
