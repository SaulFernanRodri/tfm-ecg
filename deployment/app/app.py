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
