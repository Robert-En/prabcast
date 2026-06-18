"""
Design System für PrABCast
===========================
Zentrale UI-Komponenten für konsistente Benutzeroberfläche

Dieses Modul stellt standardisierte UI-Komponenten bereit, um eine
einheitliche und professionelle Benutzeroberfläche zu gewährleisten.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional, Callable, Any


# ============================================================================
# GLOBAL PLOTLY TEMPLATE – used by all charts across the app
# ============================================================================

_CHART_COLORS = [
    "#1F77B4",  # blue
    "#E45756",  # red
    "#2CA02C",  # green
    "#7B2D8E",  # purple
    "#FF7F0E",  # orange
    "#17BECF",  # cyan
    "#D62728",  # dark red
    "#9467BD",  # medium purple
    "#8C564B",  # brown
    "#E377C2",  # pink
    "#BCBD22",  # olive
    "#1B9E77",  # teal
]

_prabcast_template = go.layout.Template()
_prabcast_template.layout = go.Layout(
    font=dict(family="Arial, sans-serif", size=13),
    colorway=_CHART_COLORS,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(l=60, r=30, t=50, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

pio.templates["prabcast"] = _prabcast_template
pio.templates.default = "prabcast"

CHART_COLORS = _CHART_COLORS  # public access


def get_chart_colors(count: int):
    """Return a color list of the requested length, cycling if needed."""
    if count <= 0:
        return []
    return [CHART_COLORS[index % len(CHART_COLORS)] for index in range(count)]


class UIComponents:
    """
    Zentrale Klasse für UI-Komponenten mit konsistenten Styles
    
    Enthält:
    - Icons für verschiedene Kontexte
    - Farbschema (Streamlit-kompatibel)
    - Standardisierte Buttons
    - Section-Headers mit optionaler Hilfe
    - Status-Indikatoren
    """
    
    # ============================================================================
    # ICONS - Konsistente Emojis für die gesamte Anwendung
    # ============================================================================
    
    ICONS = {
        # Navigation & Workflow
        'upload': '📁',
        'analysis': '📊',
        'forecast': '🎯',
        'model': '🔬',
        'data': '📈',
        'settings': '⚙️',
        'dashboard': '📉',
        
        # Status & Feedback
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'help': '❓',
        'tip': '💡',
        
        # Actions
        'play': '▶️',
        'stop': '⏹️',
        'refresh': '🔄',
        'download': '📥',
        'save': '💾',
        'delete': '🗑️',
        'back': '↩️',
        
        # Quality & Validation
        'check': '✔️',
        'cross': '✖️',
        'star': '⭐',
        'time': '⏱️',
        'calendar': '📅',
        'chart': '📊',
        
        # Special
        'rocket': '🚀',
        'fire': '🔥',
        'target': '🎯',
        'magic': '✨',
        'lock': '🔒',
        'unlock': '🔓'
    }
    
    # ============================================================================
    # COLORS - Streamlit-kompatibles Farbschema
    # ============================================================================
    
    COLORS = {
        'primary': '#FF4B4B',      # Streamlit Rot
        'success': '#21C354',      # Grün
        'warning': '#FFA500',      # Orange
        'error': '#FF4B4B',        # Rot
        'info': '#4A9EFF',         # Blau
        'neutral': '#808495',      # Grau
        'best': '#d4edda',         # Hellgrün – beste Metrik
        'worst': '#f8d7da',        # Hellrot  – schlechteste Metrik
    }
    
    # ============================================================================
    # BUTTONS - Standardisierte Button-Komponenten
    # ============================================================================
    
    @staticmethod
    def primary_button(
        label: str,
        key: Optional[str] = None,
        icon: Optional[str] = None,
        disabled: bool = False,
        use_container_width: bool = True,
        help: Optional[str] = None,
        use_icons: bool = False
    ) -> bool:
        """
        Haupt-Action-Button mit konsistentem Styling
        
        Args:
            label: Button-Text
            key: Eindeutiger Key für Streamlit
            icon: Optional - Icon-Schlüssel aus ICONS dict
            disabled: Button deaktiviert
            use_container_width: Button auf volle Breite
            help: Tooltip-Text
            use_icons: Icons anzeigen (default: False für professionelles Design)
            
        Returns:
            True wenn Button geklickt wurde
            
        Example:
            if UI.primary_button("Analyse starten", key="analysis_btn"):
                run_analysis()
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}"
        else:
            display_label = label
        
        return st.button(
            display_label,
            key=key,
            type="primary",
            disabled=disabled,
            use_container_width=use_container_width,
            help=help
        )
    
    @staticmethod
    def secondary_button(
        label: str,
        key: Optional[str] = None,
        icon: Optional[str] = None,
        disabled: bool = False,
        use_container_width: bool = False,
        help: Optional[str] = None,
        use_icons: bool = False
    ) -> bool:
        """
        Sekundärer Button für weniger wichtige Aktionen
        
        Args:
            label: Button-Text
            key: Eindeutiger Key
            icon: Optional - Icon-Schlüssel
            disabled: Button deaktiviert
            use_container_width: Button auf volle Breite
            help: Tooltip-Text
            use_icons: Icons anzeigen (default: False)
            
        Returns:
            True wenn Button geklickt wurde
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}".strip()
        else:
            display_label = label
        
        return st.button(
            display_label,
            key=key,
            type="secondary",
            disabled=disabled,
            use_container_width=use_container_width,
            help=help
        )
    
    # ============================================================================
    # HEADERS & SECTIONS
    # ============================================================================
    
    @staticmethod
    def section_header(
        title: str,
        icon: Optional[str] = None,
        help_text: Optional[str] = None,
        divider: bool = True,
        use_icons: bool = False
    ):
        """
        Standardisierter Section-Header mit optionaler Hilfe
        
        Args:
            title: Header-Text
            icon: Optional - Icon-Schlüssel aus ICONS dict (nicht verwendet wenn use_icons=False)
            help_text: Optional - Hilfetext wird als Caption unter Header angezeigt
            divider: Trennlinie unter Header anzeigen
            use_icons: Icons anzeigen (default: False)
            
        Example:
            UI.section_header("Modellauswahl", help_text="Wählen Sie ein Prognosemodell")
        """
        # Header-Text zusammenstellen
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            header_text = f"{icon_str} {title}".strip()
        else:
            header_text = title
        
        st.subheader(header_text)
        
        # Hilfetext als Caption direkt unter Header
        if help_text:
            st.caption(help_text)
        
        if divider:
            st.divider()
    
    @staticmethod
    def page_header(
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Haupt-Page-Header für Tab-Überschriften
        
        Args:
            title: Haupt-Titel
            subtitle: Optional - Untertitel/Beschreibung
            icon: Optional - Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            header_text = f"{icon_str} {title}".strip()
        else:
            header_text = title
        
        st.header(header_text)
        
        if subtitle:
            st.caption(subtitle)
        
        st.divider()
    
    # ============================================================================
    # STATUS MESSAGES
    # ============================================================================
    
    @staticmethod
    def success_message(
        message: str,
        icon: str = 'success',
        expandable_details: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Erfolgs-Nachricht mit optionalen Details
        
        Args:
            message: Haupt-Nachricht
            icon: Icon-Schlüssel (default: success, nicht verwendet wenn use_icons=False)
            expandable_details: Optional - Zusätzliche Details in Expander
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons:
            icon_str = UIComponents.ICONS.get(icon, UIComponents.ICONS['success'])
            st.success(f"{icon_str} {message}")
        else:
            st.success(message)
        
        if expandable_details:
            with st.expander("Details anzeigen"):
                st.write(expandable_details)
    
    @staticmethod
    def error_message(
        message: str,
        details: Optional[str] = None,
        show_recovery: bool = False,
        on_retry: Optional[Callable] = None,
        on_back: Optional[Callable] = None,
        use_icons: bool = False
    ):
        """
        Fehler-Nachricht mit optionalen Recovery-Optionen
        
        Args:
            message: Haupt-Fehlermeldung
            details: Optional - Technische Details
            show_recovery: Recovery-Buttons anzeigen
            on_retry: Callback für "Neu versuchen" Button
            on_back: Callback für "Zurück" Button
            use_icons: Icons anzeigen (default: False)
        """
        st.error(message)
        
        if details:
            with st.expander("Technische Details"):
                st.code(details)
        
        if show_recovery:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Zurück", key="error_back"):
                    if on_back:
                        on_back()
            with col2:
                if st.button("Neu versuchen", key="error_retry"):
                    if on_retry:
                        on_retry()
    
    @staticmethod
    def warning_message(
        message: str,
        icon: str = 'warning',
        details: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Warnung mit optionalen Details
        
        Args:
            message: Warnungs-Text
            icon: Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            details: Optional - Zusätzliche Informationen
            use_icons: Icons anzeigen (default: False)
        """
        st.warning(message)
        
        if details:
            with st.expander("Mehr Informationen"):
                st.write(details)
    
    @staticmethod
    def info_message(
        message: str,
        icon: str = 'info',
        collapsible: bool = False,
        use_icons: bool = False
    ):
        """
        Info-Nachricht
        
        Args:
            message: Info-Text
            icon: Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            collapsible: Als Expander anzeigen
            use_icons: Icons anzeigen (default: False)
        """
        if collapsible:
            with st.expander("Information"):
                st.write(message)
        else:
            st.info(message)
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def get_icon(key: str, fallback: str = '') -> str:
        """
        Hole Icon nach Schlüssel
        
        Args:
            key: Icon-Schlüssel
            fallback: Fallback wenn Key nicht existiert
            
        Returns:
            Icon-String (Emoji)
        """
        return UIComponents.ICONS.get(key, fallback)
    
    @staticmethod
    def get_color(key: str) -> str:
        """
        Hole Farbe nach Schlüssel
        
        Args:
            key: Farb-Schlüssel
            
        Returns:
            Hex-Farbcode
        """
        return UIComponents.COLORS.get(key, UIComponents.COLORS['neutral'])
    
    @staticmethod
    def status_badge(
        status: str,
        label: Optional[str] = None
    ) -> str:
        """
        Generiere Status-Badge
        
        Args:
            status: 'success', 'error', 'warning', 'info'
            label: Optional - Text neben Icon
            
        Returns:
            Formatierter Status-String
        """
        icon_map = {
            'success': UIComponents.ICONS['success'],
            'error': UIComponents.ICONS['error'],
            'warning': UIComponents.ICONS['warning'],
            'info': UIComponents.ICONS['info'],
            'check': UIComponents.ICONS['check'],
            'cross': UIComponents.ICONS['cross']
        }
        
        icon = icon_map.get(status, UIComponents.ICONS['info'])
        
        if label:
            return f"{icon} {label}"
        return icon
    
    @staticmethod
    def metric_card(
        label: str,
        value: Any,
        delta: Optional[Any] = None,
        help_text: Optional[str] = None,
        icon: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Anzeige einer Metrik mit optionalem Delta
        
        Args:
            label: Metrik-Bezeichnung
            value: Aktueller Wert
            delta: Optional - Änderung/Vergleich
            help_text: Optional - Tooltip
            icon: Optional - Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}".strip()
        else:
            display_label = label
        
        st.metric(
            label=display_label,
            value=value,
            delta=delta,
            help=help_text
        )


# Shortcut für einfacheren Zugriff
UI = UIComponents


# ============================================================================
# MODULE-LEVEL FUNCTIONS (convenience)
# ============================================================================

def get_icon(key: str, fallback: str = '') -> str:
    """Convenience function for getting icons"""
    return UI.get_icon(key, fallback)


def get_color(key: str) -> str:
    """Convenience function for getting colors"""
    return UI.get_color(key)


# ============================================================================
# SHARED METRICS EXPLAINER (used by modellvergleich_univariate_modelle.py & modellvergleich_multivariate_modelle.py)
# ============================================================================

METRICS_EXPLANATION = """
| Metrik | Bedeutung | Interpretation |
|--------|-----------|---------------|
| **MAE** | Mittlerer absoluter Fehler | Durchschnittliche Abweichung in Originaleinheiten. Kleiner = besser. |
| **RMSE** | Wurzel des mittleren quadratischen Fehlers | Bestraft große Fehler stärker. Kleiner = besser. |
| **sMAPE** | Symmetrischer mittlerer prozentualer Fehler | Relative Genauigkeit in %. Kleiner = besser. |
| **Bias** | Systematische Abweichung | Positiv → Überschätzung, Negativ → Unterschätzung. Näher an 0 = besser. |
| **Theil's U** | Vergleich mit naiver Prognose | < 1 = besser als naiv, > 1 = schlechter als naiv. |
| **Trainingszeit** | Dauer des Modelltrainings | In Sekunden. |
| **Memory** | Speicherverbrauch beim Training | In Megabyte. |

**Performance-Diagramm:** sMAPE (Y) vs. Trainingszeit (X). Punktgröße = Speicherverbrauch. Optimal: links unten, klein.
"""


# ============================================================================
# METRICS HIGHLIGHTING – colour-code best / worst per column
# ============================================================================

def highlight_best_metrics(styler, higher_is_better=None, closest_to_zero=None):
    """
    Applies green / red background to the best / worst value in each numeric
    column of a ``pd.io.formats.style.Styler`` object.

    Args:
        styler: pandas Styler (from ``df.style``)
        higher_is_better: set of column names where a higher value is better.
                  All other numeric columns default to lower-is-better.
        closest_to_zero: set of column names where the absolute value closest
                 to zero is best.
    Returns:
        The same Styler, with ``background-color`` applied.
    """
    if higher_is_better is None:
        higher_is_better = set()
    if closest_to_zero is None:
        closest_to_zero = set()

    def _color(s):
        """Return background-color styles for a Series."""
        if s.dtype.kind not in "biufc":  # not numeric
            return [""] * len(s)

        valid_values = s.dropna()
        if len(valid_values) <= 1:
            return [""] * len(s)

        if s.name in closest_to_zero:
            abs_values = s.abs()
            is_best = abs_values == abs_values.min()
            is_worst = abs_values == abs_values.max()
        else:
            is_best = s == (s.max() if s.name in higher_is_better else s.min())
            is_worst = s == (s.min() if s.name in higher_is_better else s.max())

        colors = []
        for best, worst in zip(is_best, is_worst):
            if best:
                colors.append(f"background-color: {UIComponents.COLORS['best']}")
            elif worst:
                colors.append(f"background-color: {UIComponents.COLORS['worst']}")
            else:
                colors.append("")
        return colors

    return styler.apply(_color)

