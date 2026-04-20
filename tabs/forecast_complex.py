import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataValidationError, ModelTrainingError
from setup_module.logging_config import log_data_operation, log_model_training
from sklearn.preprocessing import MinMaxScaler
from app.models_multi import *  # Use absolute imports
# ✨ UX/UI Components
from setup_module.design_system import UIComponents
from setup_module.ui_helpers import (
    safe_execute,
    export_dialog
)
from setup_module.fanchart import render_fanchart_settings_ui, apply_fanchart_to_figure

UI = UIComponents()

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    # ✨ PROFESSIONELL: Kompakte Info-Box
    UI.info_message("""
    **Komplexe Prognose:** Erstellen Sie multivariate Absatzprognosen mit externen Einflussfaktoren. 
    Nutzen Sie vortrainierte Modelle oder trainieren Sie neue. Basis sind die Daten aus der Datenanreicherung.
    """, collapsible=True)

    if 'multivariate_data' not in st.session_state:
        UserFeedback.warning("Bitte führen Sie zuerst die Datenanreicherung durch, um die benötigten Daten zu generieren.")
        return

    st.header("Komplexe Prognose")
    # Daten laden
    data = state.forecast.multivariate_data
    
    if data is None or data.empty:
        UserFeedback.error("Keine Daten für die Prognose verfügbar.")
        return
    
    # Copy to avoid modifying original
    data = data.copy()
    
    # Add expander to display the dataset
    with st.expander("Verwendeter Datensatz für die Prognose"):
        st.write("Diese Daten werden für die multivariate Prognose verwendet. Die erste Spalte enthält die Verkaufszahlen des Produkts, die weiteren Spalten enthalten die relevanten Indizes aus der Datenanreicherung.")
        st.dataframe(data)

    # Zielvariable auswählen
    target_col = data.columns[0]

    st.write(f"In der Datenanreicherung gewähltes Produkt: **{data.columns[0]}**")

    # Prüfen, ob gespeicherte Modelle verfügbar sind
    has_saved_models = 'saved_models_for_complex' in st.session_state
    
    # Wir benötigen den session_state für das Checkbox-Widget
    if 'use_saved_models_checkbox' not in st.session_state:
        st.session_state['use_saved_models_checkbox'] = True if has_saved_models else False
    
    use_saved_models = False
    
    if has_saved_models:
        saved_info = state.forecast.saved_models['info']
        saved_target = saved_info['target_col']
        saved_date = saved_info['last_training_date']
        
        if saved_target == target_col:
            UserFeedback.success(f"Vortrainierte Modelle für {target_col} gefunden! (Trainiert bis {saved_date})")
            use_saved_models = st.checkbox(
                "Vortrainierte Modelle verwenden", 
                value=st.session_state['use_saved_models_checkbox'],
                key='use_saved_models_checkbox'
            )
            # Aktualisieren des session_state mit dem aktuellen Wert
            st.session_state['use_saved_models_checkbox'] = use_saved_models
        else:
            UserFeedback.warning(f"Die gespeicherten Modelle wurden für {saved_target} trainiert, nicht für {target_col}.")
    
    # Prognosehorizont auswählen
    forecast_horizon = st.slider(
        "Wähle den Prognosehorizont (in Monaten)",
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        key='forecast_complex_horizon',
        help="Anzahl der Monate für die Zukunftsprognose"
    )

    # Modelle auswählen
    if use_saved_models:
        available_models = list(state.forecast.saved_models['models'].keys())
        default_models = available_models
    else:
        available_models = ["LSTM", "XGBoost"]
        default_models = ["LSTM"]
    
    selected_models = st.multiselect(
        "Wähle die Modelle für die Prognose",
        available_models,
        default=default_models,
        key='forecast_complex_models',
        help="Wählen Sie ein oder mehrere multivariate Modelle für die Prognose"
    )

    st.markdown("---")
    fanchart_config = render_fanchart_settings_ui(key_prefix="forecast_complex")
    
    # ✨ NEU: Professioneller Button
    if UI.primary_button("Prognose durchführen", key='forecast_complex_btn', help="Erstellt die Prognose mit den gewählten Modellen"):
        if not fanchart_config.get("is_valid", True):
            return
        # Clear previous results first
        if 'forecast_complex_results' in st.session_state:
            del st.session_state['forecast_complex_results']
        
        # Set flag to calculate forecast
        st.session_state['forecast_status'] = 'calculating'
        
        with st.spinner("Führe multivariate Prognose durch..."):
            try:
                # Daten vorbereiten
                features = data  # Zielvariable bleibt enthalten
                target = data[target_col]
                
                if use_saved_models:
                    # Verwende gespeicherte Modelle und Scaler
                    saved_models = state.forecast.saved_models['models']
                    scaler_X = state.forecast.saved_models['scalers']['X']
                    scaler_y = state.forecast.saved_models['scalers']['y']
                    seq_length = state.forecast.saved_models['info']['seq_length']
                    
                    # Normalisiere nur die Daten
                    X_scaled = scaler_X.transform(features)
                else:
                    # Neu trainieren
                    # Normalisierung
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()

                    X_scaled = scaler_X.fit_transform(features)
                    y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

                    # Sequenzen erstellen
                    seq_length = 12
                    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

                    # Gesamte Daten zum Training verwenden
                    X_train = X_seq
                    y_train = y_seq

                # Zukünftige Daten für die Prognose vorbereiten
                last_sequence = X_scaled[-seq_length:]

                forecast_results = {}

                for model_name in selected_models:
                    if use_saved_models:
                        # Verwende vortrainierte Modelle
                        model = saved_models[model_name]
                    else:
                        # Trainiere neue Modelle
                        if model_name == "LSTM":
                            model = build_lstm_model(input_shape=(seq_length, X_scaled.shape[1]))
                            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        elif model_name == "XGBoost":
                            X_train_flat = X_train.reshape(X_train.shape[0], -1)
                            model = build_xgboost_model()
                            model.fit(X_train_flat, y_train)
                        else:
                            UserFeedback.error(f"Unbekanntes Modell: {model_name}")
                            continue
                    
                    # Prognose durchführen
                    if model_name == "LSTM":
                        forecast_input = last_sequence.reshape(1, seq_length, X_scaled.shape[1])
                        forecast_scaled = []
                        for _ in range(forecast_horizon):
                            pred = model.predict(forecast_input)
                            forecast_scaled.append(pred[0, 0])
                            forecast_input = np.roll(forecast_input, -1, axis=1)
                            forecast_input[0, -1, :] = np.append(pred, forecast_input[0, -1, 1:])
                        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
                    elif model_name == "XGBoost":
                        forecast_input = last_sequence.reshape(1, -1)
                        forecast_scaled = []
                        for _ in range(forecast_horizon):
                            pred = model.predict(forecast_input)
                            forecast_scaled.append(pred[0])
                            forecast_input = np.roll(forecast_input, -1)
                            forecast_input[0, -1] = pred
                        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)

                    # Inverse Transformation
                    forecast = scaler_y.inverse_transform(forecast_scaled)

                    # Erstelle Datumsindex für Prognose
                    last_date = data.index[-1]
                    forecast_dates = pd.date_range(last_date, periods=forecast_horizon+1, freq='M')[1:]

                    forecast_series = pd.Series(forecast.flatten(), index=forecast_dates)
                    forecast_results[model_name] = forecast_series

                # Speichere die Prognoseergebnisse im Session State
                st.session_state['forecast_complex_results'] = forecast_results
                st.session_state['forecast_complex_target'] = target_col
                st.session_state['forecast_status'] = 'success'
                
                # Use rerun to refresh the page and show results
                st.rerun()

            except Exception as e:
                UserFeedback.error(f"Fehler bei der Durchführung der Prognose: {str(e)}")
                st.exception(e)
                st.session_state['forecast_status'] = 'error'
    
    # Display results section - only runs if we have results
    if 'forecast_complex_results' in st.session_state and st.session_state.get('forecast_status') == 'success':
        display_forecast_results(data, st.session_state['forecast_complex_results'], 
                               st.session_state['forecast_complex_target'], fanchart_config)

def display_forecast_results(data, forecast_results, target_col, fanchart_config):
    """Helper function to display forecast results"""
    st.subheader("Prognoseergebnisse")
    
    # Create plot
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[target_col],
        mode='lines',
        name='Historische Daten'
    ))

    # Forecasts for each model
    for name, forecast in forecast_results.items():
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name=f'Prognose ({name})'
        ))
        fig = apply_fanchart_to_figure(
            fig=fig,
            historical_data=data[target_col],
            forecast=forecast,
            config=fanchart_config,
            model_name=name
        )

    fig.update_layout(
        title=f"Prognose für {target_col}",
        xaxis_title="Datum",
        yaxis_title="Wert",
        height=600
    )

    # Display chart with unique key
    st.plotly_chart(fig, key="forecast_result_chart")

    # Display forecast data table
    if forecast_results:
        all_forecasts = pd.DataFrame(forecast_results)
        
        st.subheader("Prognosewerte")
        
        # Format table for display
        all_forecasts_styled = format_summary_table(all_forecasts, all_forecasts.columns.tolist())
        st.dataframe(all_forecasts_styled, key="forecast_table")
        
        # ✨ NEU: Export mit Metadata
        export_dialog(
            data=all_forecasts,
            filename_base="prognose_komplex",
            metadata={
                'Produkt': target_col,
                'Modelle': ', '.join(forecast_results.keys())
            }
        )

def create_sequences(X, y, seq_length=12):
    """Erstellt Sequenzen für die Zeitreihenvorhersage."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)