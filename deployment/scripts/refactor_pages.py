import re

# 1. Refactor 2_test_simulador.py
with open("Desarrollo/tfm-ecg/deployment/app/pages/2_test_simulador.py", "r") as f:
    content2 = f.read()

# Remove print CSS
content2 = re.sub(r'@media print \{.*?\n\}\n', '', content2, flags=re.DOTALL)
content2 = re.sub(r'/\* Ajustes generales de página para el PDF.*?\*/\n', '', content2, flags=re.DOTALL)

# Remove print button
content2 = re.sub(r'# ── BOTÓN IMPRIMIR INFORME CLÍNICO ────────────────────────────────────.*?\)\s*# ──────────────────────────────────────────────────────────────────────', '', content2, flags=re.DOTALL)

# Remove lead importance function
content2 = re.sub(r'def plot_lead_importance_plotly.*?return fig\n', '', content2, flags=re.DOTALL)

# Remove lead importance module calls
content2 = re.sub(r'with col_xai2:\s*run_leads = st\.checkbox\("Módulo Espacial \(Derivaciones\)", value=True\)\s*', '', content2)
content2 = re.sub(r'col_run, col_xai1, col_xai2, col_xai3 = st\.columns\(\[1\.5, 1, 1, 1\]\)', 'col_run, col_xai1, col_xai3 = st.columns([1.5, 1, 1])', content2)
content2 = re.sub(r'if run_leads:\s*with st\.spinner\("Realizando ablación espacial \(Lead Importance\)\.\.\."\):\s*importances = compute_lead_importance\(model, ecg_t, clin_t, thresholds, top_class, top_idx\)\s*st\.markdown\("<h4 style=\'color:#0f4c81;\'>Importancia de Derivaciones</h4>", unsafe_allow_html=True\)\s*st\.plotly_chart\(plot_lead_importance_plotly\(importances, top_class\), use_container_width=True\)', '', content2)

# Remove data_source logic, make it only demo
content2 = re.sub(r'with col_origen:\s*st\.markdown\("\*\*Origen del Electrocardiograma\*\*"\)\s*data_source = st\.radio\("Origen del ECG", \["Muestra Demo del Dataset", "Subir CSV Estructurado"\], label_visibility="collapsed"\)\s*if data_source == "Muestra Demo del Dataset":', 'with col_origen:\n        st.markdown("**Origen del Electrocardiograma**")\n        if True:', content2)
content2 = re.sub(r'else:\s*uploaded = st\.file_uploader\("Archivo CSV \(1000x12\)", type=\["csv"\]\)\s*if uploaded is not None:\s*import pandas as pd\s*df = pd\.read_csv\(uploaded, header=None\)\s*if df\.shape == \(1000, 12\):\s*st\.session_state\["csv_df"\] = df\.values\s*else:\s*st_blue_alert\(f"Error: El CSV debe tener forma \(1000, 12\)\. Se detectó {df\.shape}\."\)', '', content2)

content2 = re.sub(r'if data_source == "Muestra Demo del Dataset":', 'if True:', content2)
content2 = re.sub(r'else:\s*age\s*=\s*st\.number_input\("Edad \(años\)", min_value=10, max_value=110, value=55, step=None\)\s*height = st\.number_input\("Altura \(cm\)", min_value=120, max_value=220, value=170, step=None\)', '', content2)
content2 = re.sub(r'else:\s*sex\s*=\s*st\.radio\("Sexo", \["Hombre", "Mujer"\], horizontal=True\)\s*sex_v\s*=\s*0 if sex == "Hombre" else 1\s*weight = st\.number_input\("Peso \(kg\)", min_value=30, max_value=200, value=75, step=None\)', '', content2)

content2 = re.sub(r'# Procesar CSV si fue subido y se le da al botón\s*if data_source == "Subir CSV Estructurado" and analyze_btn and "csv_df" in st\.session_state:\s*st\.session_state\["ecg_raw"\]  = preprocess_ecg_single\(st\.session_state\["csv_df"\], stats\)\s*st\.session_state\["clin_raw"\] = preprocess_clinical_single\(age, sex_v, height, weight, scaler, medians\)\s*st\.session_state\["true_labels"\] = None\s*st\.session_state\["ecg_ready"\] = True\s*', '', content2)

content2 = re.sub(r'if True:\s*clin = st\.session_state\["clin_raw"\]\s*else:\s*clin = preprocess_clinical_single\(age, sex_v, height, weight, scaler, medians\)', 'clin = st.session_state["clin_raw"]', content2)

# Fix title
content2 = re.sub(r'<p style=\'text-align:center; color:#555;\'>Módulo de Diagnóstico Electrocardiográfico Multilabel</p>', "<p style='text-align:center; color:#555;'>Test de Validación (Muestras PTB-XL)</p>", content2)

with open("Desarrollo/tfm-ecg/deployment/app/pages/2_test_simulador.py", "w") as f:
    f.write(content2)


# 2. Refactor 3_simulador_csv.py
with open("Desarrollo/tfm-ecg/deployment/app/pages/3_simulador_csv.py", "r") as f:
    content3 = f.read()

# Remove print
content3 = re.sub(r'@media print \{.*?\n\}\n', '', content3, flags=re.DOTALL)
content3 = re.sub(r'/\* Ajustes generales de página para el PDF.*?\*/\n', '', content3, flags=re.DOTALL)
content3 = re.sub(r'# ── BOTÓN IMPRIMIR INFORME CLÍNICO ────────────────────────────────────.*?\)\s*# ──────────────────────────────────────────────────────────────────────', '', content3, flags=re.DOTALL)

# Remove lead importance
content3 = re.sub(r'def plot_lead_importance_plotly.*?return fig\n', '', content3, flags=re.DOTALL)
content3 = re.sub(r'with col_xai2:\s*run_leads = st\.checkbox\("Módulo Espacial \(Derivaciones\)", value=True\)\s*', '', content3)
content3 = re.sub(r'col_run, col_xai1, col_xai2, col_xai3 = st\.columns\(\[1\.5, 1, 1, 1\]\)', 'col_run, col_xai1, col_xai3 = st.columns([1.5, 1, 1])', content3)
content3 = re.sub(r'if run_leads:\s*with st\.spinner\("Realizando ablación espacial \(Lead Importance\)\.\.\."\):\s*importances = compute_lead_importance\(model, ecg_t, clin_t, thresholds, top_class, top_idx\)\s*st\.markdown\("<h4 style=\'color:#0f4c81;\'>Importancia de Derivaciones</h4>", unsafe_allow_html=True\)\s*st\.plotly_chart\(plot_lead_importance_plotly\(importances, top_class\), use_container_width=True\)', '', content3)

# Remove data_source logic, make it only CSV
content3 = re.sub(r'with col_origen:\s*st\.markdown\("\*\*Origen del Electrocardiograma\*\*"\)\s*data_source = st\.radio\("Origen del ECG", \["Muestra Demo del Dataset", "Subir CSV Estructurado"\], label_visibility="collapsed"\)\s*if data_source == "Muestra Demo del Dataset":\s*sample_idx = st\.number_input\("ID de paciente demo \(0-49\)", min_value=0, max_value=49, value=0, step=None\)\s*if st\.button\("Cargar Muestra Demo", use_container_width=True\):\s*with st\.spinner\("Cargando registro…"\):\s*test_ecg, test_clin, test_labels = load_test_samples\(50\)\s*st\.session_state\["ecg_raw"\]  = test_ecg\[sample_idx\]\s*st\.session_state\["clin_raw"\] = test_clin\[sample_idx\]\s*st\.session_state\["true_labels"\] = test_labels\[sample_idx\]\s*st\.session_state\["ecg_ready"\] = True\s*else:\s*uploaded = st\.file_uploader\("Archivo CSV \(1000x12\)", type=\["csv"\]\)\s*if uploaded is not None:\s*import pandas as pd\s*df = pd\.read_csv\(uploaded, header=None\)\s*if df\.shape == \(1000, 12\):\s*st\.session_state\["csv_df"\] = df\.values\s*else:\s*st_blue_alert\(f"Error: El CSV debe tener forma \(1000, 12\)\. Se detectó {df\.shape}\."\)', 'with col_origen:\n        st.markdown("**Subir Registro Electrocardiográfico**")\n        uploaded = st.file_uploader("Archivo CSV (1000x12)", type=["csv"])\n        if uploaded is not None:\n            import pandas as pd\n            df = pd.read_csv(uploaded, header=None)\n            if df.shape == (1000, 12):\n                st.session_state["csv_df"] = df.values\n            else:\n                st_blue_alert(f"Error: El CSV debe tener forma (1000, 12). Se detectó {df.shape}.")', content3)

content3 = re.sub(r'if data_source == "Muestra Demo del Dataset":\s*if st\.session_state\.get\("ecg_ready"\) and "clin_raw" in st\.session_state:\s*c_data = st\.session_state\["clin_raw"\]\s*unscaled = scaler\.inverse_transform\(\[\[c_data\[0\], c_data\[2\], c_data\[3\]\]\]\)\[0\]\s*st\.info\(f"\*\*Edad:\*\* \{int\(round\(unscaled\[0\]\)\)\} años"\)\s*st\.info\(f"\*\*Altura:\*\* \{int\(round\(unscaled\[1\]\)\)\} cm"\)\s*else:\s*st\.caption\("Cargue una muestra para ver los datos biométricos\."\)\s*else:\s*age\s*=\s*st\.number_input\("Edad \(años\)", min_value=10, max_value=110, value=55, step=None\)\s*height = st\.number_input\("Altura \(cm\)", min_value=120, max_value=220, value=170, step=None\)', 'age = st.number_input("Edad (años)", min_value=10, max_value=110, value=55, step=None)\n        height = st.number_input("Altura (cm)", min_value=120, max_value=220, value=170, step=None)', content3)

content3 = re.sub(r'if data_source == "Muestra Demo del Dataset":\s*if st\.session_state\.get\("ecg_ready"\) and "clin_raw" in st\.session_state:\s*c_data = st\.session_state\["clin_raw"\]\s*unscaled = scaler\.inverse_transform\(\[\[c_data\[0\], c_data\[2\], c_data\[3\]\]\]\)\[0\]\s*sex_str = "Mujer" if c_data\[1\] == 1 else "Hombre"\s*st\.info\(f"\*\*Sexo:\*\* \{sex_str\}"\)\s*st\.info\(f"\*\*Peso:\*\* \{int\(round\(unscaled\[2\]\)\)\} kg"\)\s*else:\s*sex\s*=\s*st\.radio\("Sexo", \["Hombre", "Mujer"\], horizontal=True\)\s*sex_v\s*=\s*0 if sex == "Hombre" else 1\s*weight = st\.number_input\("Peso \(kg\)", min_value=30, max_value=200, value=75, step=None\)', 'sex = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)\n        sex_v = 0 if sex == "Hombre" else 1\n        weight = st.number_input("Peso (kg)", min_value=30, max_value=200, value=75, step=None)', content3)

content3 = re.sub(r'if data_source == "Subir CSV Estructurado" and analyze_btn and "csv_df" in st\.session_state:', 'if analyze_btn and "csv_df" in st.session_state:', content3)
content3 = re.sub(r'if data_source == "Muestra Demo del Dataset":\s*clin = st\.session_state\["clin_raw"\]\s*else:\s*clin = preprocess_clinical_single\(age, sex_v, height, weight, scaler, medians\)', 'clin = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)', content3)

# Fix title
content3 = re.sub(r'<p style=\'text-align:center; color:#555;\'>Módulo de Diagnóstico Electrocardiográfico Multilabel</p>', "<p style='text-align:center; color:#555;'>Inferencia con Datos Externos (CSV)</p>", content3)

with open("Desarrollo/tfm-ecg/deployment/app/pages/3_simulador_csv.py", "w") as f:
    f.write(content3)

