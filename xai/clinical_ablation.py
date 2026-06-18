"""
Ablación de variables clínicas para interpretabilidad rápida (XAI).

Este módulo permite calcular la contribución marginal de cada variable
clínica al riesgo final de una predicción, mediante ablación contrafactual
o sustitución por la media poblacional. Es una alternativa más rápida a SHAP.
"""

import numpy as np
import tensorflow as tf

def compute_clinical_influence(
    model: tf.keras.Model, 
    ecg: np.ndarray, 
    clin: np.ndarray, 
    class_idx: int
) -> np.ndarray:
    """
    Calcula la contribución al riesgo de cada variable clínica.
    
    Para variables continuas (edad, altura, peso), se sustituyen por su valor
    medio en la población (0.0 en el espacio escalado).
    Para la variable binaria (sexo), se realiza un análisis contrafactual
    sustituyéndola por el sexo opuesto.
    
    Args:
        model: Modelo Keras.
        ecg: Array (T, 12) de señal ECG.
        clin: Array (4,) con [edad, sexo, altura, peso] escalados.
        class_idx: Índice de la clase a evaluar.
        
    Returns:
        Array (4,) con la diferencia absoluta de probabilidad al ablar la variable.
        Positivo = la variable actual aumenta el riesgo.
        Negativo = la variable actual reduce el riesgo.
    """
    ecg_t  = tf.convert_to_tensor(ecg[np.newaxis], dtype=tf.float32)
    clin_t = tf.convert_to_tensor(clin[np.newaxis], dtype=tf.float32)
    baseline = float(model([ecg_t, clin_t], training=False).numpy()[0][class_idx])
    
    influences = []
    
    # 0: age, 1: sex, 2: height, 3: weight
    for i in range(len(clin)):
        ablated = clin.copy()
        
        if i == 1:
            # Sexo: análisis contrafactual (si es 0 -> 1, si es 1 -> 0)
            # Como la variable puede venir ligeramente escalada/redondeada, usamos 1.0 - ablated[i]
            ablated[i] = 1.0 - ablated[i]
        else:
            # Variables continuas: valor medio poblacional (0.0)
            ablated[i] = 0.0
            
        abl_t = tf.convert_to_tensor(ablated[np.newaxis], dtype=tf.float32)
        p = float(model([ecg_t, abl_t], training=False).numpy()[0][class_idx])
        
        influences.append(baseline - p)
        
    return np.array(influences)
