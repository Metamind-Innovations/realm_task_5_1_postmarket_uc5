import argparse
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from utils import load_csv, store_json, get_config, pass_checks


def validate_dataframe_rows(
    df: pd.DataFrame,
    expected_values: Dict[str, Union[List[Any], List[Union[int, float]]]],
    expected_types: Dict[str, str],
) -> pd.Series:
    """Validate DataFrame rows against expected criteria.

    :param df: DataFrame to validate.
    :param expected_values: Mapping of column names to allowed categorical
        values or numeric ``[min_value, max_value]`` ranges.
    :param expected_types: Mapping of column names to expected data types.
    :return: Boolean series where ``True`` indicates valid rows.
    """

    # Start with all rows being valid
    is_valid = pd.Series(True, index=df.index)

    for column, exp_values in expected_values.items():
        if column not in df.columns:
            continue

        column_data_type = expected_types.get(column)
        if not column_data_type:
            continue

        if column_data_type == "categorical":
            # Categorical check
            is_valid &= df[column].isin(exp_values)
        elif column_data_type == "numeric":
            # Numerical check
            min_val, max_val = exp_values[0], exp_values[1]
            numeric_values = pd.to_numeric(df[column], errors="coerce")
            column_is_valid = numeric_values.between(min_val, max_val)
            is_valid &= column_is_valid

    return is_valid


def expert_knowledge_check(
    synth_data: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Union[str, List[int], Dict[int, Dict[str, Any]]]]:
    """Validate synthetic data against expert knowledge criteria.

    Applies row-by-row validation to a DataFrame using predefined expected values
    and returns a summary of valid/invalid rows along with expert knowledge information.
    Invalid rows are identified as those where is_row_invalid() returns False.

    Args:
        synth_data (pd.DataFrame): The synthetic DataFrame to validate.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Union[str, List[int], Dict[int, Dict[str, Any]]]]:
            Dictionary containing:
            - "Information": Expert knowledge description from config
            - "Valid Rows": List of row indices that passed all validations
            - "Invalid Rows": Dictionary mapping row indices to their invalid data
    """

    expected_values = config.get("expected_values", {})
    expected_types = config.get("column_types", {})

    synth_data_with_flag = synth_data.copy()

    synth_data_with_flag["is_valid"] = validate_dataframe_rows(
        df=synth_data_with_flag,
        expected_values=expected_values,
        expected_types=expected_types,
    )

    valid_rows_indices = synth_data_with_flag[
        synth_data_with_flag["is_valid"]
    ].index.tolist()

    invalid_rows_data = synth_data_with_flag[~synth_data_with_flag["is_valid"]].copy()
    invalid_rows_data.drop(columns=["is_valid"], inplace=True)
    invalid_rows_data = invalid_rows_data.to_dict("index")
    invalid_rows_data = dict(
        map(lambda item: (str(item[0]), item[1]), invalid_rows_data.items())
    )

    information = config.get("information").get("expert_knowledge")

    return {
        "Information": information,
        "Valid Rows": valid_rows_indices,
        "Invalid Rows": invalid_rows_data,
    }


def run_expert_knowledge_check(
    synth_path: Path,
    output_path: Path,
) -> None:
    """Run expert knowledge validation on synthetic data against clinical ranges.

    Args:
        synth_path (Path): Path to synthetic feature data CSV file.
        output_path (Path): Path where expert knowledge evaluation results
            will be saved.
    """

    # Load data
    synth_data = load_csv(synth_path)

    # Load config
    config = get_config()

    pass_checks(synth=synth_data, config=config)

    # Run expert analysis
    expert_knowledge_results = expert_knowledge_check(
        synth_data=synth_data, config=config
    )

    # Store results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_json(data=expert_knowledge_results, path=output_path)

    print(f"Expert Knowledge Evaluation Completed. Results saved to {output_path}")


def main() -> None:
    """Main entry point for expert knowledge evaluation of COPowereD synthetic data.

    Parses command line arguments and executes expert knowledge validation
    against clinical ranges and expected values from configuration.
    """

    parser = argparse.ArgumentParser(
        description="Run expert knowledge evaluation for COPowereD synthetic data"
    )
    parser.add_argument(
        "--synth_data", required=True, help="Path to tabular synthetic data CSV file"
    )
    parser.add_argument(
        "--output",
        default="output/expert_knowledge_evaluation_results.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Execute expert knowledge evaluation
    run_expert_knowledge_check(
        synth_path=Path(args.synth_data),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
