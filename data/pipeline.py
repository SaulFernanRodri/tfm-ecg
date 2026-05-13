"""
Módulo de pipeline de datos con tf.data.Dataset.

Responsabilidades:
- Construir pipelines tf.data.Dataset eficientes para train/val/test
- Soportar pesos de muestra (sample_weight) en el dataset de entrenamiento
- Aplicar aumentación de señal ECG durante entrenamiento (via data.augmentation)
- Configurar batching, shuffling y prefetch con AUTOTUNE
- Adaptar el formato de entrada al modelo multimodal (dos ramas nombradas)

El uso de tf.data.Dataset con prefetch asíncrono reduce el cuello de
botella I/O durante el entrenamiento al solapar el preprocesamiento
de datos con el cómputo en GPU/CPU.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from data.augmentation import make_augment_map_fn

# Buffer de shuffle para el dataset de entrenamiento.
# Usar el tamaño completo del split de train (~17 000 muestras) garantiza
# un muestreo uniforme sin sesgo de orden.
SHUFFLE_BUFFER: int = 17_500

# tf.data.AUTOTUNE delega la elección del número de workers a TensorFlow
AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(
    ecg:            np.ndarray,
    clinical:       np.ndarray,
    labels:         np.ndarray,
    batch_size:     int = 32,
    shuffle:        bool = False,
    seed:           int = 42,
    sample_weights: Optional[np.ndarray] = None,
    augment:        bool = False,
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset a partir de arrays numpy.

    El dataset produce elementos de la forma:
        ({"ecg_input": tensor, "clinical_input": tensor}, labels)
    o, si se proporcionan pesos:
        ({"ecg_input": tensor, "clinical_input": tensor}, labels, weights)

    Los nombres de clave del diccionario de entradas ("ecg_input" y
    "clinical_input") deben coincidir exactamente con los nombres de
    los objetos Input del modelo (definidos en model/fusion.py).

    Args:
        ecg:            Array (N, T, 12) de señales ECG normalizadas.
        clinical:       Array (N, 4) de variables clínicas escaladas.
        labels:         Array (N, 5) de etiquetas multilabel binarias.
        batch_size:     Tamaño del batch. Por defecto 32.
        shuffle:        Si True, aplica shuffle con buffer completo.
                        Recomendado solo para el split de train.
        seed:           Semilla del shuffle. Por defecto 42.
        sample_weights: Array opcional (N,) de pesos por muestra.
                        Si se proporciona, se incluye en el dataset.
        augment:        Si True, aplica aumentación ECG per-sample
                        (amplitude scale, noise, time shift, lead masking).
                        Solo debe activarse para el split de entrenamiento.

    Returns:
        tf.data.Dataset configurado con batch y prefetch.
    """
    inputs = {
        "ecg_input":      tf.cast(ecg,      tf.float32),
        "clinical_input": tf.cast(clinical,  tf.float32),
    }
    targets = tf.cast(labels, tf.float32)

    if sample_weights is not None:
        weights = tf.cast(sample_weights, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets, weights)
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    # Aumentación per-sample (antes del batch para aplicar aleatoriedad independiente)
    if augment:
        aug_fn = make_augment_map_fn(with_weights=(sample_weights is not None))
        dataset = dataset.map(aug_fn, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def create_all_datasets(
    train_ecg:      np.ndarray,
    train_clinical: np.ndarray,
    train_labels:   np.ndarray,
    val_ecg:        np.ndarray,
    val_clinical:   np.ndarray,
    val_labels:     np.ndarray,
    test_ecg:       np.ndarray,
    test_clinical:  np.ndarray,
    test_labels:    np.ndarray,
    batch_size:     int = 32,
    seed:           int = 42,
    train_sample_weights: Optional[np.ndarray] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Crea los tres datasets (train, val, test) del pipeline completo.

    - Train: shuffle activado, sample_weights opcionales.
    - Val / Test: sin shuffle, sin pesos.

    Args:
        train_ecg, train_clinical, train_labels:   Datos de entrenamiento.
        val_ecg, val_clinical, val_labels:          Datos de validación.
        test_ecg, test_clinical, test_labels:       Datos de test.
        batch_size:           Tamaño del batch. Por defecto 32.
        seed:                 Semilla del shuffle. Por defecto 42.
        train_sample_weights: Pesos por muestra para train (opcional).

    Returns:
        Tupla (train_ds, val_ds, test_ds).
    """
    train_ds = create_dataset(
        train_ecg, train_clinical, train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        sample_weights=train_sample_weights,
        augment=True,  # aumentación solo en train
    )
    val_ds = create_dataset(
        val_ecg, val_clinical, val_labels,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = create_dataset(
        test_ecg, test_clinical, test_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        f"[Pipeline] Batches → Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds
