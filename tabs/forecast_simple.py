# forecast_simple.py
from datetime import datetime
from typing import Dict, Any
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.logging_config import log_data_operation
from setup_module.forecast_helpers import (
    train_forecast_model,
    create_forecast_chart
)
# ✨ NEU: Model Registry statt direkter Imports
from setup_module.model_registry import get_model_registry

# ✨ NEUE UX: UI Components
from setup_module.design_system import UI
from setup_module.ui_helpers import display_model_selector_with_info, safe_execute, display_spinner, export_dialog
# ✨ SMART DEFAULTS: Intelligente Empfehlungen
from setup_module.smart_defaults import SmartDefaults
# ✨ CONTEXTUAL HELP: Hilfe-System für Produktionsplaner
from setup_module.help_system import HelpSystem, show_smart_warning
from setup_module.fanchart import render_fanchart_settings_ui, apply_fanchart_to_figure


def display_tab():
    """Tab mit verbesserter UX und Model Selector."""
    state = get_app_state()
    
    # ✨ PROFESSIONELLER: Info-Box ohne Expander
    UI.info_message("""
    **Univariate Absatzprognose:** Erstellen Sie Vorhersagen basierend auf historischen Verkaufsdaten 
    eines einzelnen Produkts. Die Prognose verwendet ausschließlich die Zeitreihe des ausgewählten Produkts.
    """)
    
    if not state.data.df is not None:
        UI.error_message(
            "Bitte laden Sie zuerst Daten hoch.",
            details="Gehen Sie zum 'Datenansicht'-Tab, um Ihre CSV-Datei hochzuladen und zu konfigurieren.",
            show_recovery=False
        )
        return
        
    UI.page_header("Einfache Absatzprognose", subtitle="Univariate Zeitreihenprognose für ein Produkt")

    # ✨ PROFESSIONELLER: Section Header
    UI.section_header("Produkt und Zeitraum", help_text="Konfigurieren Sie die Grundeinstellungen für die Prognose")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Produkt auswählen
        selected_product = st.selectbox(
            "Produkt",
            state.data.selected_products,
            key='forecast_simple_product',
            help="Wählen Sie das Produkt, für das eine Prognose erstellt werden soll"
        )
    
    # Produktdaten vorbereiten für Smart Defaults
    df = state.data.df.copy()
    df[state.data.date_column] = pd.to_datetime(df[state.data.date_column])
    df.set_index(state.data.date_column, inplace=True)
    product_data = df[selected_product].resample('M').sum()
    
    # ✨ SMART DEFAULTS: Analyse durchführen
    recommendations = SmartDefaults.get_smart_recommendations(product_data)
    recommended_horizon = recommendations['horizon']['recommended']
    max_horizon = recommendations['horizon']['max']
    
    # ✨ CONTEXTUAL HELP: Datenqualität prüfen
    quality_warnings = HelpSystem.check_data_quality(product_data)
    if quality_warnings:
        with st.expander("⚠️ Hinweise zur Datenqualität", expanded=True):
            for warning in quality_warnings:
                st.markdown(warning)
    
    with col2:
        # ✨ SMART DEFAULT: Prognosehorizont mit intelligenter Empfehlung
        col2a, col2b, col2c = st.columns([3, 1, 1])
        with col2a:
            forecast_horizon = st.slider(
                "Prognosehorizont (Monate)",
                min_value=1,
                max_value=max_horizon,
                value=recommended_horizon,
                key='forecast_simple_horizon',
                help=recommendations['horizon']['reason']
            )
        with col2b:
            st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
            if st.button("🤖", key='show_smart_recommendations', help="Zeige intelligente Empfehlungen"):
                st.session_state['show_recommendations'] = not st.session_state.get('show_recommendations', False)
        with col2c:
            st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
            # ✨ HELP BUTTON: Glossar-Popup
            if st.button("📖", key='btn_horizon_help', help="Was ist ein Prognosehorizont?"):
                st.session_state['popup_horizon_help'] = not st.session_state.get('popup_horizon_help', False)
    
    # ✨ HELP: Prognosehorizont Erklärung
    if st.session_state.get('popup_horizon_help', False):
        HelpSystem.show_glossar_popup('Prognosehorizont', use_simple=False)
    
    # ✨ CONTEXTUAL WARNING: Prognosehorizont zu lang?
    horizon_warning = HelpSystem.check_forecast_horizon(len(product_data), forecast_horizon)
    if horizon_warning:
        show_smart_warning(horizon_warning, warning_type="warning")
    
    # ✨ SMART RECOMMENDATIONS: Info-Box wenn gewünscht
    if st.session_state.get('show_recommendations', False):
        with st.expander("Intelligente Analyse & Empfehlungen", expanded=True):
            # Saisonalität
            season = recommendations['seasonality']
            if season['has_seasonality']:
                st.info(f"**Saisonalität erkannt:** {season['reason']}")
            else:
                st.info(f"**Keine Saisonalität:** {season['reason']}")
            
            with st.popover("📖 Was ist Saisonalität?"):
                entry = HelpSystem.GLOSSAR['Saisonalität']
                st.caption(entry['full_name'])
                st.markdown(entry['detailed'])
            
            # Horizon
            st.success(f"**Empfohlener Prognosehorizont:** {recommended_horizon} Monate")
            st.caption(recommendations['horizon']['reason'])
            
            # Modelle mit Glossar-Links
            models_rec = recommendations['models']
            st.markdown("**Empfohlene Modelle:**")
            for model in models_rec['primary']:
                if model in models_rec['reasons']:
                    col_m1, col_m2 = st.columns([5, 1])
                    with col_m1:
                        st.markdown(f"**{model}**: {models_rec['reasons'][model]}")
                    with col_m2:
                        if model in HelpSystem.GLOSSAR:
                            with st.popover("📖"):
                                entry = HelpSystem.GLOSSAR[model]
                                st.caption(entry['full_name'])
                                st.markdown(entry['detailed'])

    # ✨ NEUE UX: Model Selector mit Registry  
    st.markdown("---")
    
    # Help Button für Modellauswahl
    if st.button("❓ Wie wähle ich ein Modell?", key='model_selection_help', help="Hilfe zur Modellauswahl"):
        st.session_state['show_model_help'] = not st.session_state.get('show_model_help', False)
    
    if st.session_state.get('show_model_help', False):
        with st.expander("💡 Hilfe zur Modellauswahl", expanded=True):
            st.markdown(HelpSystem.explain_parameter('model_selection'))
    
    selected_model, model_metadata = display_model_selector_with_info(
        data_length=len(product_data),
        has_seasonality=recommendations['seasonality']['has_seasonality'],
        key_prefix="forecast_simple",
        default_models=recommendations['models']['primary'][:1]  # Beste Empfehlung vorauswählen
    )
    
    if not selected_model:
        UI.warning_message("Bitte wählen Sie ein Modell aus.")
        return
    
    # ✨ CONTEXTUAL WARNING: Modell-Eignung prüfen
    model_warning = HelpSystem.check_model_suitability(
        selected_model,
        len(product_data),
        recommendations['seasonality']['has_seasonality']
    )
    if model_warning:
        show_smart_warning(model_warning, warning_type="info")

    st.markdown("---")
    fanchart_config = render_fanchart_settings_ui(key_prefix="forecast_simple")

    st.markdown("---")
    
    # ✨ PROFESSIONELLER: Button mit Design System
    if UI.primary_button("Prognose erstellen", key='forecast_simple_run'):
        if not fanchart_config.get("is_valid", True):
            return
        with display_spinner("Erstelle Prognose..."):
            # ✨ SAFE EXECUTE: Error Handling
            success, result = safe_execute(
                run_forecast_simple,
                error_message="Prognose konnte nicht erstellt werden",
                show_details=True,
                show_recovery=True,
                state=state,
                selected_product=selected_product,
                forecast_horizon=forecast_horizon,
                selected_model=selected_model
            )
            
            if not success:
                return
            
            # ✨ SUCCESS FEEDBACK
            UI.success_message(
                f"Prognose für '{selected_product}' erfolgreich erstellt ({forecast_horizon} Monate)",
                expandable_details=f"Verwendetes Modell: {selected_model}"
            )
            
            # Ergebnisse anzeigen
            display_forecast_results(result, selected_product, forecast_horizon, fanchart_config)


def run_forecast_simple(state, selected_product, forecast_horizon, selected_model) -> Dict[str, Any]:
    """
    Führt die Prognose aus - separiert für besseres Error Handling
    
    Returns:
        Dict mit Prognose-Ergebnissen
    """
    # Daten vorbereiten
    df = state.data.df.copy()
    df[state.data.date_column] = pd.to_datetime(df[state.data.date_column])
    df.set_index(state.data.date_column, inplace=True)
    
    # Produktdaten extrahieren
    product_data = df[selected_product].resample('M').sum()
    
    log_data_operation(
        "Produktdaten vorbereitet",
        data_shape=(len(product_data), 1),
        product=selected_product
    )
    
    # ✨ NEU: Modell aus Registry holen
    registry = get_model_registry()
    model_class = registry.get(selected_model)
    
    # Modell trainieren
    model = train_forecast_model(model_class, product_data)
    if model is None:
        raise Exception(f"Modell {selected_model} konnte nicht trainiert werden")
    
    # Prognose erstellen
    forecast = model.predict(steps=forecast_horizon)
    
    return {
        'product_data': product_data,
        'forecast': forecast,
        'model_name': selected_model,
        'model': model,
        'model_class': model_class
    }


def display_forecast_results(result, selected_product, forecast_horizon, fanchart_config: dict)-> None:
    """
    Zeigt Prognose-Ergebnisse an
    
    Args:
        result: Dict mit Prognose-Daten
        selected_product: Produktname
        forecast_horizon: Prognosehorizont
        fanchart_config: Konfiguration für Fanchart
    """
    product_data = result['product_data']
    forecast = result['forecast']
    model_name = result['model_name']
    model_class = result.get('model_class')
    
    UI.section_header("Prognose-Ergebnisse", help_text="Visualisierung und Download der Prognosewerte")

    # Visualisierung
    forecasts = {model_name: forecast}
    fig = create_forecast_chart(
        historical_data=product_data,
        forecasts=forecasts,
        test_data=None,
        title=f"Prognose für {selected_product}",
        product_name=selected_product
    )
    fig = apply_fanchart_to_figure(
        fig=fig,
        historical_data=product_data,
        forecast=forecast,
        config=fanchart_config,
        model_name=model_name,
        model_class=model_class,
        train_model_func=train_forecast_model,
        fitted_model=result.get("model"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ✨ NEUE UX: Export Dialog
    forecasts_df = pd.DataFrame(forecasts)
    export_dialog(
        forecasts_df,
        filename_base=f"prognose_{selected_product}_{model_name}",
        metadata={
            'product': selected_product,
            'model': model_name,
            'horizon': forecast_horizon,
            'created': datetime.now().isoformat()
        }
    )

def auto_select_models(product_data):
    """Automatische Modellauswahl basierend auf Dateneigenschaften"""
    from statsmodels.tsa.stattools import adfuller, acf

    # Stationaritätstest
    adf_result = adfuller(product_data.values)
    is_stationary = adf_result[1] < 0.05
    
    # Saisonalitätserkennung
    acf_vals = acf(product_data.values, nlags=24)
    has_seasonality = any(acf_vals[12:13] > 0.3)
    
    # Länge der Zeitreihe
    series_length = len(product_data)
    
    # Volatilität
    volatility = product_data.std() / product_data.mean()
    
    # Modellvorschläge basierend auf Eigenschaften
    suggested_models = []
    
    if series_length < 24:
        suggested_models.extend(["Moving Average", "SES"])
    else:
        if is_stationary:
            suggested_models.append("ARIMA")
        else:
            suggested_models.append("SARIMA")
        
        if has_seasonality:
            suggested_models.extend(["Prophet", "Holt-Winters"])
        
        if series_length >= 50:
            if volatility > 0.5:
                suggested_models.extend(["XGBoost", "Random Forest"])
            if series_length >= 100:
                suggested_models.extend(["LSTM", "GRU"])
    
    return suggested_models