import matplotlib.pyplot as plt
import numpy as np

# 1. Definición de los datos
versiones = ['v1\n(Baseline)', 'v2\n(Focal Loss)', 'v3\n(ResNet5+SE)', 'v4\n(ASL)', 'v5\n(Augemntación)']
auc_macro = [0.9076, 0.9107, 0.9137, 0.9094, 0.9255]
sensibilidad_macro = [0.6602, 0.9389, 0.9410, 0.9364, 0.9463]

# 2. Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpio y académico
fig, ax1 = plt.subplots(figsize=(12, 6))

# Colores elegantes
color_auc = '#1f77b4'  # Azul corporativo
color_sens = '#d62728' # Rojo oscuro

# 3. Creación del gráfico de barras para el AUC (Eje Y principal)
x = np.arange(len(versiones))
width = 0.35  # Ancho de las barras

# Ajustamos la posición para que las barras queden centradas si agrupamos,
# pero aquí como usamos un segundo eje, podemos ponerlas en la misma 'x' con un offset
bar1 = ax1.bar(x - width/2, auc_macro, width, label='AUC Macro', color=color_auc, alpha=0.8, edgecolor='black')

ax1.set_ylabel('AUC Macro', color=color_auc, fontsize=12, fontweight='bold')
ax1.set_ylim(0.85, 0.95) # Ajustamos el zoom del eje Y para destacar las diferencias
ax1.tick_params(axis='y', labelcolor=color_auc)
ax1.set_xticks(x)
ax1.set_xticklabels(versiones, fontsize=11)

# 4. Creación del gráfico de líneas para la Sensibilidad (Eje Y secundario)
ax2 = ax1.twinx()  # Instanciamos un segundo eje que comparte el mismo eje X

# Dibujamos la sensibilidad como una línea con marcadores para enfatizar la evolución
line1 = ax2.plot(x + width/2, sensibilidad_macro, color=color_sens, marker='o', markersize=8, 
                 linewidth=3, label='Sensibilidad Macro')

ax2.set_ylabel('Sensibilidad Macro (Recall)', color=color_sens, fontsize=12, fontweight='bold')
ax2.set_ylim(0.60, 1.0) # Eje desde 0.60 para visualizar la caída drástica de la v1
ax2.tick_params(axis='y', labelcolor=color_sens)

# 5. Añadir la línea del Objetivo de Diseño (RNF-04)
linea_objetivo = ax2.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Objetivo Sensibilidad (≥0.90)')

# 6. Añadir las etiquetas de valor numérico exacto encima de cada punto/barra
# Para el AUC
for rect in bar1:
    height = rect.get_height()
    ax1.annotate(f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 puntos de offset vertical
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color_auc)

# Para la Sensibilidad
for i, v in enumerate(sensibilidad_macro):
    ax2.annotate(f'{v:.4f}', 
                 (x[i] + width/2, v),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', fontsize=10, fontweight='bold', color=color_sens)

# 7. Título y Leyendas
plt.title('Evolución del Rendimiento del Modelo (Estudio de Ablación v1-v5)', fontsize=14, fontweight='bold', pad=20)

# Combinar leyendas de ambos ejes
bars_lines = [bar1, line1[0], linea_objetivo]
labels = [l.get_label() for l in bars_lines]
ax1.legend(bars_lines, labels, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

# 8. Limpieza final y guardado
plt.tight_layout()
nombre_archivo = 'evolucion_modelos_tfm.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight') # Alta resolución para documentos Word/PDF
print(f"Gráfico generado y guardado exitosamente como '{nombre_archivo}'")