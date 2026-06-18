import re

for filename in ["Desarrollo/tfm-ecg/deployment/app/pages/2_test_simulador.py", "Desarrollo/tfm-ecg/deployment/app/pages/3_simulador_csv.py"]:
    with open(filename, "r") as f:
        content = f.read()

    # Fix the print button
    content = re.sub(r'# ── BOTÓN IMPRIMIR INFORME CLÍNICO ────────────────────────────────────.*?components\.html\(print_btn_html, height=70\)', '', content, flags=re.DOTALL)
    
    # Fix the tabs
    content = content.replace('tab1, tab2, tab3 = st.tabs(["Localización Temporal (Grad-CAM)", "Ablación de Derivaciones", "Análisis Contrafactual Clínico"])', 'tab1, tab3 = st.tabs(["Localización Temporal (Grad-CAM)", "Análisis Contrafactual Clínico"])')
    
    # Remove the entire `with tab2:` block
    # It starts with "with tab2:" and ends right before "with tab3:"
    content = re.sub(r'    with tab2:\n.*?(?=    with tab3:)', '', content, flags=re.DOTALL)

    with open(filename, "w") as f:
        f.write(content)

