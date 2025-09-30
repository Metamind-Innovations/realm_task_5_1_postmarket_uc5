import pandas as pd
import numpy as np
import json
from typing import Union, Optional, Any, Dict
from pathlib import Path

filepath_dir = Path(__file__).parent
CONFIG_PATH = filepath_dir / "config.json"


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (Union[str, Path]): Path to the CSV file.

    Returns:
        pd.DataFrame: Data from the CSV file.
    """

    df = pd.read_csv(file_path, sep=",")
    return df


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a Python dictionary.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy data types.
    This encoder extends the standard :class:`json.JSONEncoder` to handle
    NumPy-specific objects that are not JSON serializable by default.

    Supported conversions:
        - np.integer → int
        - np.floating → float
        - np.ndarray → list
        - np.bool_ / bool → bool
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def store_json(data: Any, path: Union[str, Path]) -> None:
    """Store data as a JSON file.

    Args:
        data (Any): The data to be serialized into JSON.
        path (Union[str, Path]): The file path where the JSON will be stored.

    Returns:
        None
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def get_config() -> Dict[str, Any]:
    """Load and return the configuration dictionary from the config file.

    Returns:
        Dict[str, Any]: Configuration dictionary containing expected values,
                       columns, data types, and analysis information.
    """

    return load_json(CONFIG_PATH)


def pass_checks(
    config: Dict,
    synth: pd.DataFrame,
    rwd: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
) -> None:
    """Validate input data and configuration for post-market evaluation.

    Args:
        config (Dict): Configuration dictionary containing expected values.
        synth (pd.DataFrame): Synthetic data.
        rwd (Optional[pd.DataFrame]): Real-world data. Defaults to None.
        target_col (Optional[str]): Target column name. Defaults to None.

    Raises:
        ValueError: If data validation fails.
    """

    # Always check synth
    if synth is None:
        raise ValueError(
            f"Synthetic data can not be None. Check again the data provided."
        )

    if synth.empty:
        raise ValueError(
            f"Synthetic data can not be empty. Check again the data provided."
        )

    # Check expert knowledge columns in synth
    expert_knowledge_columns = list(config.get("expected_values").keys())
    missing_cols_synth = set(expert_knowledge_columns) - set(synth.columns)
    if bool(missing_cols_synth):
        raise ValueError(
            f"Columns that are evaluated through expert knowledge cannot be missing. Check again the data provided."
        )

    # If adversarial evaluation data is provided, validate all components
    adversarial_data_provided = target_col is not None

    if adversarial_data_provided:
        if rwd is None:
            raise ValueError(
                f"Real World data can not be None. Check again the data provided."
            )

        if rwd.empty:
            raise ValueError(
                f"Real World data can not be empty. Check again the data provided."
            )

        if not (len(synth) == len(rwd)):
            raise ValueError(
                f"Synthetic and Real World data should have the same length. Check again the data provided."
            )

        if (target_col not in synth.columns) or (target_col not in rwd.columns):
            raise ValueError(
                f"Target column should exist in the data. Check again the data provided."
            )
