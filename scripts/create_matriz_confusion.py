import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Nombres de las clases diagnósticas
clases = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# 2. Matrices de confusión extraídas de tu imagen (valores normalizados por fila)
# Formato de cada matriz: [[TN, FP], [FN, TP]]
matrices = {
    'CD': np.array([[0.77, 0.23], [0.11, 0.89]]),
    'HYP': np.array([[0.71, 0.29], [0.10, 0.90]]),
    'MI': np.array([[0.81, 0.19], [0.10, 0.90]]),
    'NORM': np.array([[0.85, 0.15], [0.08, 0.92]]),
    'STTC': np.array([[0.81, 0.19], [0.10, 0.90]])
}

# 3. Configurar la figura para una cuadrícula de 2 filas y 3 columnas (3 arriba, 2 abajo)
# Aumentamos un poco la altura (figsize) para que las dos filas respiren bien
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Añadir un título general
fig.suptitle('Matrices de confusión por subclase diagnóstica\n(valores normalizados por fila)', 
             fontsize=16, fontweight='bold', y=0.98)

# Aplanar la matriz de ejes (que originalmente es 2x3) a una lista de 6 posiciones para iterar fácilmente
axes = axes.flatten()

# 4. Bucle para dibujar cada matriz en su posición correspondiente
for i, clase in enumerate(clases):
    ax = axes[i]
    
    # Dibujar el mapa de calor (Heatmap)
    # Usamos cmap='Blues' para mantener tu paleta de colores original
    sns.heatmap(matrices[clase], annot=True, fmt='.2f', cmap='Blues', 
                cbar=False, vmin=0, vmax=1, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    
    # Títulos y etiquetas de cada subgráfica
    ax.set_title(clase, fontweight='bold', fontsize=12, pad=10)
    
    # Configurar los ejes X e Y
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Neg pred.', 'Pos pred.'], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Neg real', 'Pos real'], va='center', rotation=90, fontsize=10)
    
    # Bordes sutiles para enmarcar la matriz
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('lightgray')

# 5. Ocultar el sexto panel (índice 5), ya que solo tenemos 5 clases
axes[5].axis('off')

# 6. Ajustar el diseño para que no se superpongan los textos
# El parámetro 'rect' deja espacio en la parte superior para el título general
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 7. Guardar y mostrar
nombre_archivo = 'matrices_confusion_3x2.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Gráfico generado y guardado exitosamente como '{nombre_archivo}'")

# plt.show() # Descomenta esta línea si ejecutas esto en un Jupyter Notebook