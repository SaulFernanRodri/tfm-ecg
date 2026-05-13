---
title: ECG Diagnosis AI
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ECG Diagnosis AI 🫀

Demo de diagnóstico multilabel de ECG basado en Deep Learning con explicabilidad (XAI).

## Modelo

**ResNet1D-v5** entrenado en el dataset público [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (21 837 ECGs, 100 Hz).

Arquitectura: 5 bloques ResNet 1D con Squeeze-and-Excitation + fusión con variables clínicas (edad, sexo, altura, peso).

| Métrica        | Valor  |
|----------------|--------|
| AUC macro      | 0.9255 |
| Sensibilidad   | 0.9463 |
| F1 macro       | 0.8025 |

### Clases detectadas

| Código | Descripción              |
|--------|--------------------------|
| CD     | Trastorno de Conducción  |
| HYP    | Hipertrofia              |
| MI     | Infarto de Miocardio     |
| NORM   | ECG Normal               |
| STTC   | Cambios ST/T             |

## Explicabilidad (XAI)

- **Grad-CAM**: segmentos de la señal ECG coloreados por intensidad de activación (azul → rojo)
- **Lead Importance**: qué derivación aporta más a cada diagnóstico (ablation por derivación)

## Cómo usar

1. Selecciona una muestra del conjunto de test de PTB-XL (50 muestras pre-exportadas)
2. O sube tu propio ECG como CSV (1000 filas × 12 columnas, una por derivación, sin cabecera)
3. Ajusta los datos del paciente en el sidebar
4. Pulsa **▶ Analizar ECG**

## Aviso

> Este sistema es un prototipo de investigación. **No debe usarse para diagnóstico médico real.**

## Créditos

Trabajo de Fin de Máster — Dataset: [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
