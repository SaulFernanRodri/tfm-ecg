import sys
from pathlib import Path

# Garantizar que deployment/app/ está en sys.path antes de cargar cualquier
# página, para que 'from app_utils.ui import ...' funcione en Streamlit Cloud y HF.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

# Configuración global de la página
st.set_page_config(
    page_title="Sistema de Apoyo al Diagnóstico Electrocardiográfico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definir las páginas disponibles
pagina_inicio = st.Page("pages/1_inicio.py", title="Panel Institucional", default=True)
pagina_simulador_demo = st.Page("pages/2_test_simulador.py", title="Test Simulador (Demo)")
pagina_simulador_csv = st.Page("pages/3_simulador_csv.py", title="Simulador Clínico (CSV)")

# Configurar la navegación lateral
pg = st.navigation([pagina_inicio, pagina_simulador_demo, pagina_simulador_csv])

# Ejecutar la página seleccionada
pg.run()
