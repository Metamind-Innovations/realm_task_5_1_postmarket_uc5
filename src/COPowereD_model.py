import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class COPowereDWrapper:
    """Wrapper class for the Dockerized COPowereD COPD triage model."""

    MODEL_COLUMNS = [
        "sexe",
        "age",
        "baseline_height",
        "baseline_weight",
        "baseline_bmi",
        "baseline_copd",
        "baseline_heartRate",
        "baseline_spo2",
        "symp_worsening",
        "symp_breath",
        "symp_cough",
        "symp_sputum",
        "heartRate",
        "spo2",
    ]
    CLASS_NAMES = {0: "notNeedMedicalAttention", 1: "needMedicalAttention"}
    # Insert image to be used
    DEFAULT_IMAGE = "<docker_image>"
    DEFAULT_JAR_PATH = Path("/app/sparkServer-assembly-1.2.0.jar")
    MODEL_MAIN_CLASS = "MLProjects.bpco.triage.models.CopdComunicare_001"

    def __init__(
            self,
            image: Optional[str] = None,
            threshold: float = 0.5,
            feature_names: Optional[list] = None,
            one_dim_preds: bool = False,
            docker_executable: str = "docker",
            execution_mode: Optional[str] = None,
            jar_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the Dockerized COPowereD model wrapper.

        :param image: Docker image name. Defaults to the
            ``COPOWERED_MODEL_IMAGE`` environment variable or
            image pulled from an image repository.
        :param threshold: Classification threshold.
        :param feature_names: Feature names used when callers provide NumPy
            arrays instead of pandas DataFrames.
        :param one_dim_preds: Return only positive-class probabilities when
            ``True``.
        :param docker_executable: Docker command executable.
        :param execution_mode: ``docker`` to run the model image through Docker,
            or ``local`` to execute the model jar available in the current
            container.
        :param jar_path: Path to the COPowereD model jar for local execution.
        """
        self.image = image or os.getenv("COPOWERED_MODEL_IMAGE", self.DEFAULT_IMAGE)
        self.threshold = threshold
        self.feature_names = feature_names
        self.one_dim_preds = one_dim_preds
        self.docker_executable = docker_executable
        self.jar_path = Path(
            jar_path
            or os.getenv("COPOWERED_MODEL_JAR_PATH", str(self.DEFAULT_JAR_PATH))
        )
        self.execution_mode = (
            execution_mode
            or os.getenv("COPOWERED_MODEL_EXECUTION_MODE")
            or ("local" if self.jar_path.exists() else "docker")
        )
        self._last_probabilities = None
        self._last_predictions = None
        self._last_response = None

    def predict_proba(
            self,
            data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict class probabilities for input tabular data.

        The model expects ``data.csv`` in its input folder and writes
        ``result.csv`` with a single ``proba`` column.

        :param data: DataFrame or array with model input columns.
        :return: Probability array with shape ``(n_samples, 2)`` unless
            ``one_dim_preds`` is ``True``.
        :raises ValueError: If the input data or model output is inconsistent.
        :raises RuntimeError: If model execution fails.
        """

        if isinstance(data, np.ndarray):
            if not self.feature_names:
                raise ValueError(
                    "Input data is np.ndarray. Feature names should be passed."
                )
            data = pd.DataFrame(data=data, columns=self.feature_names)
        else:
            data = data.copy()

        missing_columns = [
            column for column in self.MODEL_COLUMNS if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                "Input data is missing columns required by the Dockerized model: "
                f"{', '.join(missing_columns)}"
            )

        model_data = data.loc[:, self.MODEL_COLUMNS]
        n_samples = len(model_data)
        if n_samples == 0:
            raise ValueError("Input data can not be empty.")

        last_error = None
        positive_probabilities = None

        with tempfile.TemporaryDirectory(prefix="copowered_model_") as tmp_dir:
            data_dir = Path(tmp_dir).resolve()
            input_dir = data_dir / "data"
            output_dir = data_dir / "result"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_csv = input_dir / "data.csv"
            result_csv = output_dir / "result.csv"
            model_data.to_csv(input_csv, index=False)

            if self.execution_mode == "local":
                if not self.jar_path.exists():
                    raise RuntimeError(
                        "COPowereD model jar not found at "
                        f"{self.jar_path}."
                    )

                command = [
                    "java",
                    "-cp",
                    str(self.jar_path),
                    self.MODEL_MAIN_CLASS,
                    "--input_folder",
                    str(input_dir),
                    "--output_folder",
                    str(output_dir),
                ]
                cwd = self.jar_path.parent
                log_config = cwd / "log4j.properties"
                log_config.write_text(
                    "\n".join(
                        [
                            "log4j.rootLogger=ERROR, stdout",
                            "log4j.appender.stdout="
                            "org.apache.log4j.ConsoleAppender",
                            "log4j.appender.stdout.Target=System.out",
                            "log4j.appender.stdout.layout="
                            "org.apache.log4j.PatternLayout",
                        ]
                    ),
                    encoding="utf-8",
                )
            elif self.execution_mode == "docker":
                result_csv = output_dir / "predictions.csv"
                command = [
                    self.docker_executable,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "sh",
                    "-v",
                    f"{data_dir.as_posix()}:/app/data",
                    self.image,
                    "-c",
                    (
                        "cd /app && "
                        "printf '%s\\n' "
                        "'log4j.rootLogger=ERROR, stdout' "
                        "'log4j.appender.stdout="
                        "org.apache.log4j.ConsoleAppender' "
                        "'log4j.appender.stdout.Target=System.out' "
                        "'log4j.appender.stdout.layout="
                        "org.apache.log4j.PatternLayout' "
                        "> log4j.properties && "
                        "java -cp /app/sparkServer-assembly-1.2.0.jar "
                        f"{self.MODEL_MAIN_CLASS} "
                        "--input_folder /app/data/data "
                        "--output_folder /app/data/result && "
                        "if [ -d /app/data/result/result.csv ]; then "
                        "cat /app/data/result/result.csv/part-* "
                        "> /app/data/result/predictions.csv; "
                        "else cp /app/data/result/result.csv "
                        "/app/data/result/predictions.csv; fi && "
                        "chmod a+r /app/data/result/predictions.csv && "
                        "printf '\\n__COPOWERED_RESULT_BEGIN__\\n' && "
                        "cat /app/data/result/predictions.csv && "
                        "printf '\\n__COPOWERED_RESULT_END__\\n'"
                    ),
                ]
                cwd = None
            else:
                raise ValueError(
                    "execution_mode must be either 'docker' or 'local'."
                )

            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=cwd,
                text=True,
            )

            if completed.returncode == 0 and result_csv.exists():
                if result_csv.is_dir():
                    result_files = sorted(result_csv.glob("part-*"))
                    if not result_files:
                        raise ValueError(
                            "COPowereD model output directory does not "
                            "contain a prediction part file."
                        )
                    result_path = result_files[0]
                else:
                    result_path = result_csv

                try:
                    result_df = pd.read_csv(result_path)
                except PermissionError:
                    start_marker = "__COPOWERED_RESULT_BEGIN__"
                    end_marker = "__COPOWERED_RESULT_END__"
                    start_index = completed.stdout.find(start_marker)
                    end_index = completed.stdout.find(end_marker)
                    if start_index == -1 or end_index == -1:
                        raise
                    csv_text = completed.stdout[
                        start_index + len(start_marker):end_index
                    ].strip()
                    result_df = pd.read_csv(io.StringIO(csv_text))
                if "proba" not in result_df.columns:
                    raise ValueError(
                        "Dockerized model output must contain a 'proba' column."
                    )
                if len(result_df) != len(model_data):
                    raise ValueError(
                        "Dockerized model output row count does not match "
                        "the input row count."
                    )

                positive_probabilities = result_df["proba"].to_numpy(
                    dtype=float
                )
                if (
                    not np.isfinite(positive_probabilities).all()
                    or (positive_probabilities < 0).any()
                    or (positive_probabilities > 1).any()
                ):
                    raise ValueError(
                        "Dockerized model probabilities must be finite "
                        "values between 0 and 1."
                    )

                self._last_response = result_df.copy()

            else:
                last_error = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "COPowereD model did not create result.csv."
                )

        if last_error is not None:
            raise RuntimeError(
                "COPowereD model failed: "
                f"{last_error}"
            )

        all_probabilities = np.column_stack(
            [1.0 - positive_probabilities, positive_probabilities]
        )

        if self.one_dim_preds:
            self._last_probabilities = positive_probabilities
            return positive_probabilities

        self._last_probabilities = all_probabilities
        return all_probabilities

    def predict(
            self,
            data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict binary class labels for input tabular data.

        :param data: DataFrame or array with model input columns.
        :return: Array of binary class labels.
        """
        probabilities = self.predict_proba(
            data=data,
        )
        positive_probabilities = (
            probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        )
        predictions = (positive_probabilities > self.threshold).astype(int)
        self._last_predictions = predictions

        return predictions

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame to the Docker model input column order.

        :param df: DataFrame with model input columns.
        :return: DataFrame with columns ordered for the Dockerized model.
        """
        missing_columns = [
            column for column in self.MODEL_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Input data is missing columns required by the Dockerized model: "
                f"{', '.join(missing_columns)}"
            )

        return df.loc[:, self.MODEL_COLUMNS].copy()

    def get_last_response(self) -> Optional[pd.DataFrame]:
        """Get the raw ``result.csv`` DataFrame from the last prediction.

        :return: Last model output DataFrame, or ``None`` if no predictions
            have been made.
        """
        if self._last_response is None:
            return None

        return self._last_response.copy()

    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class indices to class names.

        :return: Dictionary mapping class indices to descriptive names.
        """
        return self.CLASS_NAMES.copy()
