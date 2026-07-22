import streamlit as st
import pandas as pd
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_ASSETS  = _APP_DIR / "assets"

# No es necesario st.set_page_config aquí, ya se define en el app.py principal.

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #0f4c81; font-family: sans-serif;">Sistema de Apoyo a la Decisión Clínica (CDSS)</h1>
    <h3 style="color: #555; font-weight: 300;">Diagnóstico Electrocardiográfico Multimodal Basado en Deep Learning</h3>
</div>
<hr style="border-color: #ccc; margin-bottom: 2rem;">
""", unsafe_allow_html=True)

# 1. Resumen del Proyecto
st.markdown("""
### Resumen del Proyecto
Este Sistema de Apoyo a la Decisión Clínica (CDSS) está diseñado para asistir al personal médico en el **triaje automatizado y priorización de urgencias cardiológicas**. 
El sistema analiza señales electrocardiográficas (ECG) de 12 derivaciones de 10 segundos, integrando información biométrica del paciente para emitir un diagnóstico probabilístico en cinco súper-clases clínicas.
La herramienta no reemplaza el criterio médico, sino que actúa como una capa de seguridad para evitar falsos negativos en escenarios de alta carga asistencial.
""")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Columnas para Dataset y Arquitectura
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    #### Ficha Técnica del Dataset
    El sistema ha sido entrenado y validado utilizando el registro **PTB-XL v1.0.3**.
    
    * **Volumen:** 21.837 registros de ECG.
    * **Características:** 10 segundos de duración, 12 derivaciones estándar.
    * **Frecuencia de Muestreo:** 100 Hz.
    * **Distribución Diagnóstica:**
      * **NORM (Normal):** ~41%
      * **HYP (Hipertrofia):** ~12%
      * **Otras patologías:** CD, MI, STTC.
      
    *Justificación Clínica:* Existe un desbalanceo natural en la prevalencia de las patologías respecto a las trazas normales. Este desequilibrio se ha preservado deliberadamente en el entrenamiento para reflejar fielmente la prevalencia del mundo real, aplicando técnicas de compensación matemática en la función de pérdida.*
    """)

with col2:
    st.markdown("""
    #### Ficha de la Arquitectura
    El modelo subyacente es una **Arquitectura Multimodal de Fusión Tardía (Late Fusion)** que procesa paralelamente datos temporales y tabulares.
    
    * **Rama Temporal (ECG):**
      * Red Convolucional **ResNet1D** estructurada en 5 bloques profundos.
      * Módulos de atención **Squeeze-and-Excitation (SE)** para recalibrar los mapas de características a nivel de canal, priorizando los segmentos de la señal más informativos.
    * **Rama Biométrica (Tabular):**
      * **Perceptrón Multicapa (MLP)** para procesar Edad, Sexo, Altura y Peso del paciente.
    * **Función de Pérdida:**
      * **Asymmetric Loss (ASL)** calibrada dinámicamente con `gamma_neg=4`, `gamma_pos=0` y `clip=0.05`. Esta formulación mitiga el impacto del desbalanceo masivo de clases asimétricas.
    """)

st.markdown("<br>", unsafe_allow_html=True)

# 3. Diagramas de Arquitectura
st.markdown("#### Diagrama Estructural del Modelo")

_arch_general  = _ASSETS / "arquitectura_general.png"
_arch_detallada = _ASSETS / "arquitectura_detallada.png"

if _arch_general.exists() or _arch_detallada.exists():
    _cols = st.columns(2, gap="large") if (_arch_general.exists() and _arch_detallada.exists()) else [st, st]
    if _arch_general.exists():
        with _cols[0]:
            st.markdown("**Vista general del pipeline multimodal**")
            st.image(str(_arch_general), width='stretch')
    if _arch_detallada.exists():
        with _cols[1]:
            st.markdown("**Detalle de capas (ResNet1D + MLP)**")
            st.image(str(_arch_detallada), width='stretch')
else:
    st.info(
        "Guarda los diagramas de arquitectura en `deployment/app/assets/` con los nombres:\n"
        "- `arquitectura_general.png` — pipeline multimodal completo\n"
        "- `arquitectura_detallada.png` — detalle de capas ResNet1D + MLP",
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Tabla de Rendimiento Oficial (v6.1)
st.markdown("#### Rendimiento Oficial del Modelo")
st.markdown("""
Las métricas presentadas a continuación corresponden a la evaluación exhaustiva sobre el conjunto de test independiente. 
Se ha priorizado estratégicamente la **Sensibilidad (Recall)** superior al 90% para cumplir con el requisito clínico de triaje: **minimizar los falsos negativos en escenarios patológicos críticos.**
""")

# Crear un DataFrame estático con las métricas
metrics_data = {
    "Clase Diagnóstica": [
        "CD (Trastorno de Conducción)",
        "HYP (Hipertrofia)",
        "MI (Infarto de Miocardio)",
        "NORM (ECG Normal)",
        "STTC (Cambios ST/T)",
        "**MACRO MEDIA**"
    ],
    "AUC-ROC": ["0.9153", "0.8986", "0.9337", "0.9525", "0.9274", "**0.9255**"],
    "Sensibilidad (Recall)": ["0.8871", "0.9046", "0.9036", "0.9170", "0.9021", "**0.9028**"],
    "F1-Score": ["0.6667", "0.4501", "0.7282", "0.8704", "0.7170", "**0.6864**"],
    "Umbral de Decisión Clínico": ["~0.46", "~0.42", "~0.53", "~0.65", "~0.47", "-"]
}

df_metrics = pd.DataFrame(metrics_data)

# Estilizado de la tabla con Markdown/HTML
st.markdown("""
<style>
    .metric-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        font-size: 14px;
        margin-top: 10px;
    }
    .metric-table th {
        background-color: #f8f9fa;
        color: #0f4c81;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #dee2e6;
    }
    .metric-table td {
        padding: 12px;
        border-bottom: 1px solid #dee2e6;
        color: #333;
    }
    .metric-table tr:hover {
        background-color: #f1f3f5;
    }
    .metric-table tr:last-child td {
        font-weight: bold;
        background-color: #f8f9fa;
        border-top: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Renderizar tabla en HTML manual para mejor control visual estricto
html_table = "<table class='metric-table'><thead><tr>"
for col in df_metrics.columns:
    html_table += f"<th>{col}</th>"
html_table += "</tr></thead><tbody>"

for index, row in df_metrics.iterrows():
    html_table += "<tr>"
    for item in row:
        html_table += f"<td>{item}</td>"
    html_table += "</tr>"

html_table += "</tbody></table>"

st.markdown(html_table, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Métricas Adicionales Globales: Precisión Macro: 0.5733 | Especificidad Macro: 0.8528")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Curvas ROC
st.markdown("#### Curvas ROC por Clase Diagnóstica")
st.markdown(
    "Las curvas ROC muestran la capacidad discriminativa del modelo para cada superclase. "
    "El modelo alcanza un **AUC macro de 0.925**, con NORM como la clase mejor separada (AUC 0.95) "
    "y HYP como la más exigente (AUC 0.90)."
)

_roc_path = _ASSETS / "roc_por_clase_v5.png"
if _roc_path.exists():
    col_roc, _ = st.columns([2, 1])
    with col_roc:
        st.image(str(_roc_path), width='stretch')
else:
    st.warning("Gráfica ROC no encontrada en assets/roc_por_clase_v5.png")
