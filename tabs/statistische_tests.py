# tabs/statistische_tests.py
from setup_module.helpers import *
from setup_module.session_state import get_app_state
from setup_module.error_handler import ErrorHandler, UserFeedback
from setup_module.exceptions import DataValidationError, ModelTrainingError
from setup_module.logging_config import log_data_operation, log_model_training
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import streamlit as st

def display_tab():
    """Tab mit strukturiertem Session State Management."""
    state = get_app_state()
    
    # Collapsible text section with information for both tests
    with st.expander("Informationen zu den statistischen Tests"):
        st.write("""
            In diesem Tab können Sie zwei wichtige statistische Tests zur Überprüfung der Stationarität Ihrer 
            Zeitreihen durchführen. Stationarität ist eine Eigenschaft, bei der die statistischen Eigenschaften einer 
            Zeitreihe über die Zeit konstant bleiben. Stationarität ist eine Voraussetzung für viele Zeitreihenmodelle.

            ### ADF-Test (Augmented Dickey-Fuller-Test)
            Der ADF-Test prüft die Nullhypothese, dass eine Zeitreihe eine Einheitswurzel besitzt, was auf Nicht-Stationarität hinweist. Eine stationäre Zeitreihe hat konstante Mittelwerte und Varianzen über die Zeit hinweg, was für viele Zeitreihenmodelle eine Voraussetzung ist.

            **Anwendungsbeispiel:** Der ADF-Test kann verwendet werden, um festzustellen, ob die monatlichen Verkaufszahlen eines Produkts saisonal stabil sind oder ob ein Trend vorliegt, der eine Differenzierung erfordert.

            ### KPSS-Test (Kwiatkowski-Phillips-Schmidt-Shin-Test)
            Der KPSS-Test prüft die Nullhypothese, dass eine Zeitreihe stationär ist. Im Gegensatz zum ADF-Test testet der KPSS-Test direkt auf Stationarität, wodurch er eine ergänzende Perspektive bietet.

            **Anwendungsbeispiel:** Der KPSS-Test kann eingesetzt werden, um zu bestätigen, ob die identifizierte Stationarität der Verkaufszahlen stabil bleibt oder ob externe Faktoren zu Veränderungen führen.
            
            **Wie funktioniert der Prozess in diesem Tab?**

            **1. Produkt Auswahl**: Wählen Sie einfach ein Produkt aus, das Sie testen möchten und die Tests werden sofort durchgeführt und angezeigt.

            **Hinweis:** 
            - Ein signifikanter p-Wert beim ADF-Test deutet auf Stationarität hin, was bedeutet, dass die Nullhypothese der Nicht-Stationarität abgelehnt wird.
            - Ein signifikanter p-Wert beim KPSS-Test weist ebenfalls auf eine Ablehnung der Nullhypothese hin, was in diesem Fall bedeutet, dass die Zeitreihe nicht stationär ist.

            **Zusätzliche Informationen:**
            - **Stationarität:** Eine Zeitreihe ist stationär, wenn ihre statistischen Eigenschaften wie Mittelwert, Varianz und Autokorrelation im Zeitverlauf konstant bleiben.
            - **Nicht-Stationarität:** Eine Zeitreihe ist nicht stationär, wenn sie Trends, saisonale Muster oder andere Strukturen aufweist, die ihre statistischen Eigenschaften im Zeitverlauf verändern.
        """)

    # Produktauswahl
    selected_product_st_test = st.selectbox("Wähle ein Produkt für die Tests", state.data.selected_products)
    df_for_test = state.data.df[selected_product_st_test]

    # Analyse in zwei Spalten
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ADF-Test (Augmented Dickey-Fuller-Test)")
        try:
            adf_result = adfuller(df_for_test)
            display_adf_results(adf_result)
        except Exception as e:
            UserFeedback.error(f"Fehler beim Durchführen des ADF-Tests: {e}")

    with col2:
        st.subheader("KPSS-Test (Kwiatkowski-Phillips-Schmidt-Shin-Test)")
        try:
            kpss_result = kpss(df_for_test, regression='ct')
            display_kpss_results(kpss_result)
        except Exception as e:
            UserFeedback.error(f"Fehler beim Durchführen des KPSS-Tests: {e}")

def display_adf_results(adf_result):
    statistic, p_value, lags, n_obs, critical_values, icbest = adf_result
    st.write("**Ergebnisse des ADF-Tests**")
    test_results_df = pd.DataFrame({
        'Metrik': ['ADF Test Statistik', 'P-Wert', 'Anzahl der Lags', 'Beobachtungen'],
        'Wert': [f"{statistic:.4f}", f"{p_value:.4f}", str(lags), str(n_obs)]
    })
    st.dataframe(test_results_df, hide_index=True)
    critical_values_df = pd.DataFrame(list(critical_values.items()), columns=['Signifikanzniveau', 'Kritischer Wert'])
    st.write("**Kritische Werte:**")
    st.dataframe(critical_values_df, hide_index=True)
    interpret_p_value(p_value, test_type="ADF-Test")

def display_kpss_results(kpss_result):
    statistic, p_value, lags, critical_values = kpss_result
    st.write("**Ergebnisse des KPSS-Tests**")
    test_results_df = pd.DataFrame({
        'Metrik': ['KPSS Test Statistik', 'P-Wert', 'Anzahl der Lags'],
        'Wert': [f"{statistic:.4f}", f"{p_value:.4f}", str(lags)]
    })
    st.dataframe(test_results_df, hide_index=True)
    critical_values_df = pd.DataFrame(list(critical_values.items()), columns=['Signifikanzniveau', 'Kritischer Wert'])
    st.write("**Kritische Werte:**")
    st.dataframe(critical_values_df, hide_index=True)
    interpret_p_value(p_value, test_type="KPSS-Test")

def interpret_p_value(p_value, test_type="Test"):
    """
    Interpretiert den p-Wert und gibt eine entsprechende Nachricht aus.

    Parameters:
    - p_value (float): Der p-Wert des Tests.
    - test_type (str): Der Typ des Tests (optional, Standard: "Test").
    """
    if p_value < 0.05:
        
        if test_type == "ADF-Test":
            UserFeedback.success(f"P-Wert ist klein (weniger als 0.05). Die Nullhypothese des {test_type} wird abgelehnt.")
            UserFeedback.info("Das bedeutet, dass die Zeitreihe stationär ist.")
        elif test_type == "KPSS-Test":
            UserFeedback.warning(f"P-Wert ist klein (weniger als 0.05). Die Nullhypothese des {test_type} wird abgelehnt.")
            UserFeedback.info("Das bedeutet, dass die Zeitreihe nicht stationär ist.")
    else:
        
        if test_type == "ADF-Test":
            UserFeedback.warning(f"P-Wert ist groß (mehr als 0.05). Die Nullhypothese des {test_type} kann nicht abgelehnt werden.")
            UserFeedback.info("Das bedeutet, dass die Zeitreihe nicht stationär ist.")
        elif test_type == "KPSS-Test":
            UserFeedback.success(f"P-Wert ist groß (mehr als 0.05). Die Nullhypothese des {test_type} kann nicht abgelehnt werden.")
            UserFeedback.info("Das bedeutet, dass die Zeitreihe stationär ist.")