"""
Utilidades para el tracking de experimentos con MLflow.

Encapsula toda la lógica de MLflow para mantener los módulos
de entrenamiento y evaluación limpios de dependencias directas.

Uso básico:
    from utils.mlflow_logger import MLflowLogger

    logger = MLflowLogger(experiment_name="TFM_ECG", run_name="v3", version="0.2.0")
    with logger:
        logger.log_params({...})
        model.fit(..., callbacks=[logger.keras_callback()])
        logger.log_metrics({...})
        logger.log_artifacts_dir(plots_dir)
        logger.register_model(model, "ECG_Multimodal")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.keras
import tensorflow as tf

# Directorio donde MLflow guarda los runs localmente
MLFLOW_TRACKING_URI = "Desarrollo/tfm-ecg/mlruns"


class MLflowLogger:
    """
    Wrapper sobre MLflow para el pipeline TFM ECG.

    Gestiona el ciclo de vida de un run: inicio, logging de parámetros,
    métricas por época (vía callback de Keras), métricas finales,
    artefactos y registro del modelo.

    Args:
        experiment_name: Nombre del experimento MLflow. Por defecto "TFM_ECG".
        run_name:        Nombre descriptivo del run (ej. "v3_resnet5_se").
        version:         Versión del modelo para el Model Registry (ej. "0.2.0").
        tags:            Diccionario de tags adicionales.
    """

    def __init__(
        self,
        experiment_name: str = "TFM_ECG",
        run_name:        str = "run",
        version:         str = "0.0.0",
        tags:            Optional[Dict[str, str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name        = run_name
        self.version         = version
        self.tags            = tags or {}
        self._run            = None

        # Configurar URI de tracking local
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowLogger":
        all_tags = {"version": self.version, **self.tags}
        self._run = mlflow.start_run(run_name=self.run_name, tags=all_tags)
        print(f"[MLflow] Run iniciado: {self.run_name} (id={self._run.info.run_id[:8]}...)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            mlflow.end_run(status="FAILED")
            print(f"[MLflow] Run finalizado con error: {exc_val}")
        else:
            mlflow.end_run(status="FINISHED")
            print(f"[MLflow] Run finalizado correctamente → {self.run_name}")
        return False  # no suprimir excepciones

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Registra hiperparámetros del experimento."""
        mlflow.log_params(params)
        print(f"[MLflow] {len(params)} parámetros registrados.")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Registra métricas finales (o de una época concreta)."""
        mlflow.log_metrics(metrics, step=step)

    def log_metrics_from_history(self, history: Dict[str, list]) -> None:
        """
        Registra el historial de entrenamiento de Keras por época.

        Convierte el diccionario history.history en métricas por paso
        para que MLflow muestre las curvas de entrenamiento.

        Args:
            history: Diccionario {metric_name: [val_epoch1, val_epoch2, ...]}.
        """
        n_epochs = len(next(iter(history.values())))
        for epoch in range(n_epochs):
            epoch_metrics = {
                key: float(vals[epoch])
                for key, vals in history.items()
            }
            mlflow.log_metrics(epoch_metrics, step=epoch + 1)
        print(f"[MLflow] Historial de {n_epochs} épocas registrado.")

    def log_metrics_from_eval(
        self,
        metrics:          Dict,
        metrics_baseline: Optional[Dict] = None,
    ) -> None:
        """
        Registra las métricas de evaluación en test.

        Registra métricas con prefijo 'test_opt_' para umbrales óptimos
        y 'test_base_' para threshold=0.5 (si se proporciona).

        Args:
            metrics:          Diccionario completo de métricas (umbral óptimo).
            metrics_baseline: Diccionario de métricas con thr=0.5 (opcional).
        """
        def _flat(m: Dict, prefix: str) -> Dict[str, float]:
            out: Dict[str, float] = {}
            out[f"{prefix}auc_macro"]         = m["auc_roc"]["macro"]
            out[f"{prefix}f1_macro"]          = m["f1_score"]["macro"]
            out[f"{prefix}sensitivity_macro"] = m["sensitivity_recall"]["macro"]
            out[f"{prefix}specificity_macro"] = m["specificity"]["macro"]
            if "precision" in m:
                out[f"{prefix}precision_macro"] = m["precision"]["macro"]
            # Por clase
            for cls, val in m["auc_roc"]["per_class"].items():
                if val is not None:
                    out[f"{prefix}auc_{cls}"] = val
            for cls, val in m["f1_score"]["per_class"].items():
                out[f"{prefix}f1_{cls}"] = val
            for cls, val in m["sensitivity_recall"]["per_class"].items():
                out[f"{prefix}sensitivity_{cls}"] = val
            if "precision" in m:
                for cls, val in m["precision"]["per_class"].items():
                    out[f"{prefix}precision_{cls}"] = val
            return out

        opt_metrics = _flat(metrics, "test_opt_")
        mlflow.log_metrics(opt_metrics)

        if metrics_baseline is not None:
            base_metrics = _flat(metrics_baseline, "test_base_")
            mlflow.log_metrics(base_metrics)

        print(f"[MLflow] Métricas de evaluación registradas "
              f"(AUC={metrics['auc_roc']['macro']:.4f}, "
              f"Sens={metrics['sensitivity_recall']['macro']:.4f}).")

    def log_artifact(self, file_path: str) -> None:
        """Registra un fichero individual como artefacto."""
        mlflow.log_artifact(file_path)

    def log_artifacts_dir(self, directory: str) -> None:
        """Registra todos los ficheros de un directorio como artefactos."""
        path = Path(directory)
        if path.exists() and path.is_dir():
            mlflow.log_artifacts(str(path))
            n = len(list(path.iterdir()))
            print(f"[MLflow] {n} artefactos registrados desde {path}.")
        else:
            print(f"[MLflow] WARN: directorio no encontrado: {directory}")

    def log_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Registra los umbrales óptimos por clase como parámetros."""
        thr_params = {f"thr_{cls}": val for cls, val in thresholds.items()}
        mlflow.log_params(thr_params)

    # ------------------------------------------------------------------
    # Registro del modelo
    # ------------------------------------------------------------------

    def register_model(
        self,
        model:      tf.keras.Model,
        model_name: str = "ECG_Multimodal",
    ) -> None:
        """
        Guarda el modelo Keras en MLflow y lo registra en el Model Registry.

        El modelo queda asociado al run activo y se registra con el nombre
        `model_name` en el registry. La versión del registro es gestionada
        automáticamente por MLflow (incremental), pero el tag 'version'
        del run indica la versión semántica del proyecto.

        Args:
            model:      Modelo Keras entrenado.
            model_name: Nombre en el Model Registry. Por defecto "ECG_Multimodal".
        """
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        print(f"[MLflow] Modelo registrado en registry como '{model_name}'.")

    # ------------------------------------------------------------------
    # Keras callback para logging por época
    # ------------------------------------------------------------------

    def keras_callback(self) -> tf.keras.callbacks.Callback:
        """
        Devuelve un callback de Keras que registra métricas por época en MLflow.

        Returns:
            Callback de Keras listo para pasar a model.fit(callbacks=[...]).
        """
        class _MLflowEpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metrics(
                        {k: float(v) for k, v in logs.items()},
                        step=epoch + 1,
                    )

        return _MLflowEpochCallback()

    # ------------------------------------------------------------------
    # Propiedad run_id
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        """Devuelve el run_id del run activo, o None si no está iniciado."""
        return self._run.info.run_id if self._run else None
