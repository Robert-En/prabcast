"""
Tab modules for PrABCast application.
Contains all UI components and tab logic.
"""

from .upload import display_tab as upload
from .aggregation import display_tab as aggregation
from .produktverteilung import display_tab as produktverteilung
from .rohdaten import display_tab as rohdaten
from .zerlegung import display_tab as zerlegung
from .abcxyz import display_tab as abcxyz
from .statistische_tests import display_tab as statistische_tests
from .modellvergleich_univariate_modelle import display_tab as forecast
from .modellvergleich_datenanreicherung import display_tab as advanced_forecast
from .modellvergleich_multivariate_modelle import display_tab as multivariate_forecast
from .prognose_univariate_prognose import display_tab as forecast_simple
from .prognose_multivariate_prognose import display_tab as forecast_complex
from .glossar import display_tab as glossar