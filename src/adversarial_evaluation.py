import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import load_csv, store_json, pass_checks, get_config
from COPowereD_model import COPowereDWrapper


def calculate_classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall,
                         and f1_score metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics


def combine_metrics_results(
        synth_metrics: Dict[str, float], rwd_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """Combines metrics results into structured dictionary with rwd,
    synthetic, and difference.

    Args:
        synth_metrics (Dict[str, float]): Metrics calculated on synthetic data.
        rwd_metrics (Dict[str, float]): Metrics calculated on real-world data.

    Returns:
        Dict[str, Dict[str, float]]: Formatted metrics with rwd, synthetic,
            and difference values.
    """

    combined_results = {}

    for metric_name in synth_metrics.keys():
        if metric_name in rwd_metrics:
            combined_results[metric_name] = {
                "rwd": rwd_metrics[metric_name],
                "synthetic": synth_metrics[metric_name],
                "difference": f"{abs(synth_metrics[metric_name] - rwd_metrics[metric_name])*100}pp",
            }

    return combined_results


def adversarial_evaluation(
        X_synth: pd.DataFrame,
        X_rwd: pd.DataFrame,
        y_synth: np.ndarray,
        y_rwd: np.ndarray,
        config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Perform adversarial evaluation comparing synthetic and real-world data.
    Evaluates the COPowereD model's performance on both synthetic and real-world
    datasets.

    Args:
        X_synth (pd.DataFrame): Synthetic patient data features.
        X_rwd (pd.DataFrame): Real-world data (RWD) patient features.
        y_synth (np.ndarray): True labels for synthetic data.
        y_rwd (np.ndarray): True labels for real-world data.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary containing information,
                            synthetic metrics, rwd metrics and comparison.
    """

    # Initialize model and results
    model = COPowereDWrapper()
    evaluation_results = {}

    # Generate predictions
    y_pred_synth = model.predict(X_synth)
    y_pred_rwd = model.predict(X_rwd)

    # Calculate metrics for synthetic amd rwd data
    synth_metrics = calculate_classification_metrics(
        y_true=y_synth, y_pred=y_pred_synth
    )
    rwd_metrics = calculate_classification_metrics(y_true=y_rwd, y_pred=y_pred_rwd)

    # Add information and combine metrics
    evaluation_results["information"] = config.get("information").get(
        "adversarial_evaluation"
    )
    combined_metrics = combine_metrics_results(synth_metrics, rwd_metrics)
    evaluation_results.update(combined_metrics)

    return evaluation_results


def run_adversarial_evaluation(
        synth_path: Path,
        rwd_path: Path,
        output_path: Path,
) -> None:
    """Run adversarial evaluation comparing synthetic vs real-world data performance.

    Args:
        synth_path (Path): Path to synthetic data CSV file.
        rwd_path (Path): Path to real-world data CSV file.
        output_path (Path): Path where adversarial evaluation results will be saved.
    """

    # Load data
    synth_data = load_csv(synth_path)
    rwd_data = load_csv(rwd_path)

    # Load config
    config = get_config()
    target = config.get("target_column")

    pass_checks(synth=synth_data, rwd=rwd_data, config=config, adversarial_checks=True)

    # Construct X and y data
    y_synth = synth_data[target].to_numpy()
    synth_data.drop(columns=[target], inplace=True)

    y_rwd = rwd_data[target].to_numpy()
    rwd_data.drop(columns=[target], inplace=True)

    # Run adversarial evaluation analysis
    adversarial_evaluation_results = adversarial_evaluation(
        X_synth=synth_data,
        X_rwd=rwd_data,
        y_synth=y_synth,
        y_rwd=y_rwd,
        config=config,
    )

    # Store results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_json(data=adversarial_evaluation_results, path=output_path)

    print(f"Adversarial Evaluation Completed. Results saved to {output_path}")


def main() -> None:
    """Run adversarial evaluation from the command line.

    Parses command line arguments and executes adversarial evaluation comparing
    model performance on synthetic and real-world datasets using the Dockerized model.
    """

    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation for COPowereD synthetic data"
    )
    parser.add_argument(
        "--synth_data", required=True, help="Path to tabular synthetic data CSV file"
    )

    parser.add_argument(
        "--rwd_data",
        required=True,
        help="Path to tabular RWD data CSV file",
    )

    parser.add_argument(
        "--output",
        default="output/adversarial_evaluation_results.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Execute adversarial evaluation
    run_adversarial_evaluation(
        synth_path=Path(args.synth_data),
        rwd_path=Path(args.rwd_data),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
