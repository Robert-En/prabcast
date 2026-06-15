"""
UI Helper Functions für PrABCast
=================================
Wiederverwendbare UI-Komponenten für Validation, Model Selection, Progress und Quality

Dieses Modul stellt spezialisierte UI-Komponenten bereit:
- Error Handling mit Recovery-Optionen
- Validation Results Dashboard
- Model Selector mit Registry-Integration
- Data Quality Dashboard
- Progress Indicators
- Export Dialogs
"""

import streamlit as st
import pandas as pd
import time
import logging
from typing import Optional, Callable, Any, List, Dict, Tuple
from datetime import datetime

# Import Design System
from setup_module.design_system import UI

# Import Data Validation
from setup_module.data_validator import ValidationResult, DataQualityReport

# Import Model Registry
from setup_module.model_registry import get_model_registry, ModelCategory

# Setup Logger
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR HANDLING
# ============================================================================

def safe_execute(
    func: Callable,
    error_message: str = "Ein Fehler ist aufgetreten",
    success_message: Optional[str] = None,
    show_details: bool = True,
    show_recovery: bool = True,
    on_retry: Optional[Callable] = None,
    on_back: Optional[Callable] = None,
    **kwargs
) -> Tuple[bool, Any]:
    """
    Führt Funktion mit User-Friendly Error Handling aus
    
    Args:
        func: Auszuführende Funktion
        error_message: User-freundliche Fehlermeldung
        success_message: Optional - Erfolgsmeldung nach Ausführung
        show_details: Technische Details anzeigen
        show_recovery: Recovery-Buttons anzeigen
        on_retry: Callback für "Neu versuchen"
        on_back: Callback für "Zurück"
        **kwargs: Arguments für func
        
    Returns:
        (success: bool, result: Any)
        
    Example:
        success, result = safe_execute(
            run_forecast,
            error_message="Prognose konnte nicht erstellt werden",
            success_message="Prognose erfolgreich erstellt!",
            model=model,
            data=data
        )
        if success:
            st.write(result)
    """
    try:
        result = func(**kwargs)
        
        # Erfolgsmeldung anzeigen falls angegeben
        if success_message:
            st.success(success_message)
        
        return True, result
        
    except Exception as e:
        # Log für Developer
        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
        
        # User-friendly Error Display
        st.error(f"**{error_message}**")
        
        # Details optional anzeigen
        if show_details:
            with st.expander("Technische Details (für Support)"):
                st.code(str(e))
                st.caption(f"Fehlertyp: {type(e).__name__}")
        
        # Lösungsvorschläge
        st.info("""
        **Mögliche Lösungen:**
        1. Überprüfen Sie Ihre Eingabedaten
        2. Stellen Sie sicher, dass alle erforderlichen Felder ausgefüllt sind
        3. Versuchen Sie einen Neustart der Seite
        4. Kontaktieren Sie den Support, falls das Problem weiterhin besteht
        """)
        
        # Recovery Options
        if show_recovery:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Zurück", key=f"error_back_{func.__name__}"):
                    if on_back:
                        on_back()
                    else:
                        st.rerun()
            with col2:
                if st.button("Neu versuchen", key=f"error_retry_{func.__name__}"):
                    if on_retry:
                        on_retry()
                    else:
                        st.rerun()
        
        return False, None


# ============================================================================
# PROGRESS INDICATORS
# ============================================================================

def run_with_progress(
    tasks: List[Callable],
    task_names: List[str],
    show_eta: bool = False,
    fail_fast: bool = True
) -> List[Any]:
    """
    Führt Tasks mit konsistentem Progress-Feedback aus
    
    Args:
        tasks: Liste von auszuführenden Funktionen
        task_names: Namen der Tasks für Anzeige
        show_eta: Geschätzte Restzeit anzeigen
        fail_fast: Bei Task-Fehler sofort mit Original-Exception abbrechen
        
    Returns:
        Liste der Ergebnisse
        
    Example:
        results = run_with_progress(
            [train_model1, train_model2, create_viz],
            ["Modell 1 trainieren", "Modell 2 trainieren", "Visualisierung"]
        )
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty() if show_eta else None
    
    results = []
    start_time = time.time()
    
    for idx, (task, name) in enumerate(zip(tasks, task_names)):
        # Status anzeigen
        status_text.text(f"Schritt {idx + 1}/{len(tasks)}: {name}...")
        
        # ETA berechnen
        if show_eta and idx > 0:
            elapsed = time.time() - start_time
            avg_time_per_task = elapsed / idx
            remaining_tasks = len(tasks) - idx
            eta_seconds = avg_time_per_task * remaining_tasks
            eta_text.caption(f"Geschätzte Restzeit: {int(eta_seconds)}s")
        
        # Task ausführen
        try:
            result = task()
            results.append(result)
        except Exception as e:
            logger.error(f"Error in task '{name}': {str(e)}", exc_info=True)
            if fail_fast:
                raise RuntimeError(f"Fehler in Task '{name}': {e}") from e
            results.append(None)
        
        # Progress aktualisieren
        progress_bar.progress((idx + 1) / len(tasks))
    
    # Cleanup
    status_text.text("Fertig!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    if eta_text:
        eta_text.empty()
    
    return results


def display_spinner(message: str = "Bitte warten..."):
    """
    Context Manager für Spinner mit konsistenter Message
    
    Example:
        with display_spinner("Berechne Prognose..."):
            run_forecast()
    """
    return st.spinner(message)


# ============================================================================
# VALIDATION RESULTS DASHBOARD
# ============================================================================

def display_validation_results(
    result: ValidationResult,
    show_metadata: bool = True,
    expanded: bool = False
) -> None:
    """
    Zeigt Validierungsergebnisse in übersichtlichem Dashboard
    
    Args:
        result: ValidationResult Objekt
        show_metadata: Metadata anzeigen
        expanded: Details standardmäßig aufgeklappt
        
    Example:
        validator = DataValidator()
        result = validator.validate_uploaded_file(df)
        display_validation_results(result)
    """
    # Header mit Overall Status
    if result.errors:
        st.error(f"**Validierung fehlgeschlagen:** {len(result.errors)} kritische Fehler gefunden")
        overall_status = "error"
    elif result.warnings:
        st.warning(f"**Validierung mit Warnungen:** {len(result.warnings)} Warnung(en) vorhanden")
        overall_status = "warning"
    else:
        st.success("**Validierung erfolgreich:** Alle Prüfungen bestanden")
        overall_status = "success"
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fehler", len(result.errors))
    with col2:
        st.metric("Warnungen", len(result.warnings))
    with col3:
        st.metric("Hinweise", len(result.info))
    
    # Details in Tabs
    if result.errors or result.warnings or result.info:
        tab_error, tab_warning, tab_info = st.tabs([
            f"Fehler ({len(result.errors)})",
            f"Warnungen ({len(result.warnings)})",
            f"Hinweise ({len(result.info)})"
        ])
        
        with tab_error:
            if result.errors:
                for idx, error in enumerate(result.errors, 1):
                    # Handle both dict and string formats
                    if isinstance(error, dict):
                        st.error(f"**#{idx}: {error.get('type', 'Fehler')}**")
                        st.write(error.get('message', ''))
                        if 'fix' in error:
                            st.info(f"**Lösung:** {error['fix']}")
                    else:
                        st.error(f"**#{idx}:** {error}")
                    st.divider()
            else:
                st.success("Keine Fehler gefunden")
        
        with tab_warning:
            if result.warnings:
                for idx, warning in enumerate(result.warnings, 1):
                    # Handle both dict and string formats
                    if isinstance(warning, dict):
                        st.warning(f"**#{idx}: {warning.get('type', 'Warnung')}**")
                        st.write(warning.get('message', ''))
                        if 'suggestion' in warning:
                            st.info(f"**Empfehlung:** {warning['suggestion']}")
                    else:
                        st.warning(f"**#{idx}:** {warning}")
                    st.divider()
            else:
                st.success("Keine Warnungen")
        
        with tab_info:
            if result.info:
                for idx, info in enumerate(result.info, 1):
                    # Handle both dict and string formats
                    if isinstance(info, dict):
                        st.info(f"**#{idx}: {info.get('type', 'Information')}**")
                        st.write(info.get('message', ''))
                    else:
                        st.info(f"**#{idx}:** {info}")
                    st.divider()
            else:
                st.info("Keine zusätzlichen Informationen")
    
    # Metadata
    if show_metadata and result.metadata:
        with st.expander("Metadaten", expanded=expanded):
            st.json(result.metadata)


# ============================================================================
# DATA QUALITY DASHBOARD
# ============================================================================

def display_quality_dashboard(
    report: DataQualityReport,
    show_trends: bool = False
) -> None:
    """
    Visuelles Dashboard für Datenqualität
    
    Args:
        report: DataQualityReport Objekt
        show_trends: Trend-Verlauf anzeigen (wenn Historie vorhanden)
        
    Example:
        validator = DataValidator()
        report = validator.assess_data_quality(df, product_col)
        display_quality_dashboard(report)
    """
    UI.section_header("Datenqualität", help_text="Bewertung der Datenqualität basierend auf verschiedenen Metriken")
    
    # Berechne Durchschnittswerte aus den Dictionaries
    avg_nan_pct = sum(report.missing_values_pct.values()) / len(report.missing_values_pct) if report.missing_values_pct else 0.0
    total_outliers = sum(report.outliers_count.values()) if report.outliers_count else 0
    num_time_gaps = len(report.time_gaps) if report.time_gaps else 0
    
    # Header mit Score
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Score mit Delta (wenn Historie vorhanden)
        delta = None
        if show_trends and 'quality_history' in st.session_state:
            if len(st.session_state.quality_history) > 1:
                previous_score = st.session_state.quality_history[-2]['score']
                delta = report.quality_score - previous_score
        
        st.metric(
            "Qualitätsscore",
            f"{report.quality_score:.1f}/100",
            delta=f"{delta:+.1f}" if delta else None,
            help="Gesamtbewertung basierend auf NaN-Anteil, Ausreißern, Varianz und Zeitlücken"
        )
        
        # Visual Score Bar
        score_color = "green" if report.quality_score >= 80 else "orange" if report.quality_score >= 60 else "red"
        st.progress(report.quality_score / 100)
    
    with col2:
        st.metric(
            "NaN-Anteil",
            f"{avg_nan_pct:.1f}%",
            help="Durchschnittlicher Prozentsatz fehlender Werte"
        )
    
    with col3:
        st.metric(
            "Ausreißer",
            total_outliers,
            help="Gesamtanzahl erkannter Ausreißer (außerhalb 3σ)"
        )
    
    with col4:
        st.metric(
            "Zeitlücken",
            num_time_gaps,
            help="Anzahl fehlender Zeitperioden"
        )
    
    # Detaillierte Metriken
    with st.expander("Detaillierte Qualitätsmetriken"):
        # Zeige Details zu missing values per Spalte
        if report.missing_values_pct:
            st.markdown("**Fehlende Werte pro Spalte:**")
            mv_data = {
                'Spalte': list(report.missing_values_pct.keys()),
                'NaN %': [f"{v:.2f}%" for v in report.missing_values_pct.values()]
            }
            st.dataframe(pd.DataFrame(mv_data), use_container_width=True, hide_index=True)
        
        # Zeige Details zu Ausreißern per Spalte
        if report.outliers_count:
            st.markdown("**Ausreißer pro Spalte:**")
            out_data = {
                'Spalte': list(report.outliers_count.keys()),
                'Anzahl': list(report.outliers_count.values())
            }
            st.dataframe(pd.DataFrame(out_data), use_container_width=True, hide_index=True)
        
        # Zeige Zeitlücken
        if report.time_gaps:
            st.markdown(f"**Zeitlücken:** {len(report.time_gaps)} gefunden")
            for i, (start, end) in enumerate(report.time_gaps[:5], 1):  # Max 5 anzeigen
                st.text(f"  {i}. {start.date()} bis {end.date()}")
            if len(report.time_gaps) > 5:
                st.text(f"  ... und {len(report.time_gaps) - 5} weitere")
        
        # Seasonality Info
        if report.seasonality_detected:
            st.markdown(f"**Saisonalität erkannt:** Periode {report.seasonality_detected}")
    
    # Trend-Tracking (optional)
    if show_trends:
        track_quality_trend(report)


def track_quality_trend(report: DataQualityReport):
    """
    Speichert Quality Score Historie und zeigt Trend
    
    Args:
        report: Aktueller DataQualityReport
    """
    from datetime import datetime
    
    # Initialize History
    if 'quality_history' not in st.session_state:
        st.session_state.quality_history = []
    
    # Add current score
    st.session_state.quality_history.append({
        'timestamp': datetime.now(),
        'score': report.quality_score
    })
    
    # Limit history to last 10 entries
    if len(st.session_state.quality_history) > 10:
        st.session_state.quality_history = st.session_state.quality_history[-10:]
    
    # Show trend if we have multiple datapoints
    if len(st.session_state.quality_history) > 1:
        with st.expander("Qualitätsverlauf"):
            history_df = pd.DataFrame(st.session_state.quality_history)
            st.line_chart(history_df.set_index('timestamp'))


def get_quality_status(value: float, metric_type: str) -> str:
    """
    Ermittelt Status-Text basierend auf Metrik-Wert
    
    Args:
        value: Metrik-Wert
        metric_type: 'nan', 'outliers', 'variance', 'gaps'
        
    Returns:
        Status-String: "Gut", "Akzeptabel", "Kritisch"
    """
    if metric_type == 'nan':
        if value < 5:
            return "Gut"
        elif value < 20:
            return "Akzeptabel"
        else:
            return "Kritisch"
    elif metric_type == 'outliers':
        if value < 3:
            return "Gut"
        elif value < 10:
            return "Akzeptabel"
        else:
            return "Kritisch"
    elif metric_type == 'variance':
        if value > 0.1:
            return "Gut"
        elif value > 0.01:
            return "Akzeptabel"
        else:
            return "Kritisch"
    elif metric_type == 'gaps':
        if value == 0:
            return "Gut"
        elif value < 3:
            return "Akzeptabel"
        else:
            return "Kritisch"
    return "Unbekannt"


# ============================================================================
# MODEL SELECTOR WITH REGISTRY INTEGRATION
# ============================================================================

def display_model_selector_with_info(
    data_length: int = 100,
    is_stationary: Optional[bool] = None,
    has_seasonality: Optional[bool] = None,
    category_filter: Optional[str] = None,
    key_prefix: str = "model_selector",
    multi_select: bool = False,
    default_models: Optional[List[str]] = None
) -> Tuple[str, Dict]:
    """
    Zeigt Model-Auswahl mit Kategorien, Anforderungen und Empfehlungen
    
    Args:
        data_length: Länge des Datensatzes (default: 100)
        is_stationary: Ob Daten stationär sind
        has_seasonality: Ob Daten Saisonalität haben
        category_filter: Optional - Filtere nach Kategorie
        key_prefix: Prefix für Streamlit Keys
        multi_select: Wenn True, mehrere Modelle auswählbar (default: False)
        default_models: Standard-Auswahl bei multi_select (default: None)
        
    Returns:
        Wenn multi_select=False: (selected_model_name: str, model_metadata: dict)
        Wenn multi_select=True: (list of model_names, dict of metadatas)
        
    Example:
        # Single Select
        model_name, metadata = display_model_selector_with_info(
            data_length=len(data),
            has_seasonality=True
        )
        
        # Multi Select
        model_names, metadatas = display_model_selector_with_info(
            data_length=len(data),
            multi_select=True,
            default_models=["ARIMA", "Prophet"]
        )
    """
    registry = get_model_registry()
    
    UI.section_header("Modellauswahl", help_text="Wählen Sie ein Prognosemodell basierend auf Ihren Daten")
    
    # Category Filter
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category_options = ["Alle Kategorien", "Statistische Modelle", "Machine Learning", "Deep Learning", "Naive Methoden"]
        category_map = {
            "Alle Kategorien": None,
            "Statistische Modelle": ModelCategory.STATISTICAL,
            "Machine Learning": ModelCategory.MACHINE_LEARNING,
            "Deep Learning": ModelCategory.DEEP_LEARNING,
            "Naive Methoden": ModelCategory.NAIVE
        }
        
        selected_category_name = st.selectbox(
            "Modell-Kategorie",
            category_options,
            key=f"{key_prefix}_category",
            help="Filtere Modelle nach Typ"
        )
        
        selected_category = category_map[selected_category_name]
    
    # Get suitable models
    suitable_models = registry.get_suitable_models(
        data_points=data_length,
        is_stationary=is_stationary,
        has_seasonality=has_seasonality
    )
    
    # Get all models (or filtered by category)
    if selected_category:
        all_models = registry.get_by_category(selected_category)
    else:
        all_models = registry.get_all_names()
    
    # Model Selection
    with col2:
        show_only_suitable = st.checkbox(
            "Nur passende Modelle",
            value=False,
            key=f"{key_prefix}_filter",
            help="Zeige nur Modelle, die für Ihre Daten geeignet sind"
        )
    
    models_to_show = suitable_models if show_only_suitable else all_models
    
    # Format function to highlight recommended models
    def format_model(name):
        if name in suitable_models:
            return f"{name} ⭐"
        return name
    
    # Model Selection - Single or Multi
    if multi_select:
        # Multi-Select mit default_models
        default = default_models if default_models else []
        selected_models = st.multiselect(
            "Modelle",
            models_to_show,
            default=[m for m in default if m in models_to_show],
            format_func=format_model,
            key=f"{key_prefix}_models",
            help="⭐ = Empfohlen für Ihre Daten. Wählen Sie mehrere Modelle zum Vergleichen."
        )
        
        # Show info for all selected models
        if selected_models:
            st.markdown(f"**{len(selected_models)} Modelle ausgewählt**")
            
            with st.expander("Details zu ausgewählten Modellen", expanded=False):
                for model_name in selected_models:
                    metadata = registry.get_metadata(model_name)
                    
                    st.markdown(f"### {model_name}")
                    st.markdown(f"{metadata.description}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Kategorie:** {metadata.category.value}")
                        st.markdown(f"**Min. Datenpunkte:** {metadata.min_data_points}")
                    with col2:
                        st.markdown(f"**Stationarität:** {'Erforderlich' if metadata.requires_stationarity else 'Optional'}")
                        st.markdown(f"**Saisonalität:** {'Unterstützt' if metadata.supports_seasonality else 'Nicht unterstützt'}")
                    
                    if model_name in suitable_models:
                        st.success("⭐ Empfohlen")
                    
                    st.markdown("---")
            
            # Return list of models and dict of metadatas
            metadatas = {name: registry.get_metadata(name).__dict__ for name in selected_models}
            return selected_models, metadatas
        else:
            return [], {}
    
    else:
        # Single Select
        selected_model = st.selectbox(
            "Modell",
            models_to_show,
            format_func=format_model,
            key=f"{key_prefix}_model",
            help="⭐ = Empfohlen für Ihre Daten"
        )
        
        # Model Info Card
        if selected_model:
            metadata = registry.get_metadata(selected_model)
            
            with st.expander(f"Details zu {selected_model}", expanded=True):
                st.markdown(f"**Beschreibung:** {metadata.description}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Kategorie:** {metadata.category.value}")
                    st.markdown(f"**Min. Datenpunkte:** {metadata.min_data_points}")
                
                with col2:
                    st.markdown(f"**Stationarität erforderlich:** {'Ja' if metadata.requires_stationarity else 'Nein'}")
                    st.markdown(f"**Unterstützt Saisonalität:** {'Ja' if metadata.supports_seasonality else 'Nein'}")
                    st.markdown(f"**Benötigt lange Historie:** {'Ja' if metadata.requires_long_history else 'Nein'}")
                    st.markdown(f"**Probabilistisches Modell:** {'Ja' if metadata.is_probabilistic else 'Nein'}")
                
                # Recommendation
                if selected_model in suitable_models:
                    st.success("**Empfohlen** für Ihre Daten basierend auf Länge und Eigenschaften")
                else:
                    st.warning("Dieses Modell erfüllt möglicherweise nicht alle Anforderungen Ihrer Daten")
                
                # Default Parameters
                show_default_params = False
                if show_default_params:
                    if metadata.default_params:
                        with st.container(border=True):
                            st.json(metadata.default_params)
            
            return selected_model, metadata.__dict__
        
        return None, {}
    
    return None, {}


# ============================================================================
# EXPORT DIALOG
# ============================================================================

def export_dialog(
    data: pd.DataFrame,
    filename_base: str,
    include_metadata: bool = True,
    metadata: Optional[Dict] = None
) -> None:
    """
    Standardisierter Export-Dialog mit Format-Optionen
    
    Args:
        data: DataFrame zum Export
        filename_base: Basis-Dateiname (ohne Extension)
        include_metadata: Metadaten einschließen
        metadata: Optional - Dict mit Metadaten
        
    Example:
        export_dialog(
            forecast_df,
            "prognose_produkt_A",
            metadata={'model': 'ARIMA', 'date': datetime.now()}
        )
    """
    # Generate unique key from filename_base
    key_prefix = filename_base.replace(" ", "_").replace("-", "_")
    
    with st.expander("Ergebnisse exportieren"):
        col1, col2 = st.columns(2)
        
        with col1:
            format_choice = st.radio(
                "Export-Format",
                ["CSV", "Excel"],
                help="Wählen Sie das gewünschte Dateiformat",
                key=f"export_format_{key_prefix}"
            )
        
        with col2:
            add_timestamp = st.checkbox(
                "Zeitstempel im Dateinamen",
                value=True,
                help="Fügt Datum und Uhrzeit zum Dateinamen hinzu",
                key=f"export_timestamp_{key_prefix}"
            )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
        filename = f"{filename_base}_{timestamp}" if timestamp else filename_base
        
        # Create download
        if format_choice == "CSV":
            # Add metadata as comment if requested
            if include_metadata and metadata:
                # Create metadata string
                meta_str = "\n".join([f"# {k}: {v}" for k, v in metadata.items()])
                csv_data = meta_str + "\n" + data.to_csv(index=False)
            else:
                csv_data = data.to_csv(index=False)
            
            st.download_button(
                label="CSV herunterladen",
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"download_csv_{key_prefix}"
            )
        
        elif format_choice == "Excel":
            # Convert to Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Daten', index=False)
                
                # Add metadata sheet if requested
                if include_metadata and metadata:
                    meta_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Wert'])
                    meta_df.to_excel(writer, sheet_name='Metadaten', index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="Excel herunterladen",
                data=excel_data,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"download_excel_{key_prefix}"
            )
