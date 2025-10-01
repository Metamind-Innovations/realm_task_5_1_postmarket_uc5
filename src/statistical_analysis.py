import pandas as pd
from typing import Dict, Any, Tuple, List
from pathlib import Path
import argparse

from utils import load_csv, store_json, pass_checks, get_config


def check_missing_values(df: pd.DataFrame) -> Tuple[int, List[int], Dict[str, int]]:
    """Check for missing values in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to analyze for missing values.

    Returns:
        Tuple[int, List[int], Dict[str, int]]:
            - Total number of missing values across entire dataframe
            - List of row indices containing missing values
            - Dictionary with missing value counts per column
    """

    # has the shape of the df
    missing_values_flag = df.isnull()

    missing_counts_per_column = missing_values_flag.sum()
    total_missing_values = missing_counts_per_column.sum()

    rows_with_missing_indices = df[missing_values_flag.any(axis=1)].index.tolist()

    # convert to dict format
    missing_counts_per_column = missing_counts_per_column.to_dict()

    return total_missing_values, rows_with_missing_indices, missing_counts_per_column


def check_column_existence(
    df: pd.DataFrame, expected_columns: List[str]
) -> Tuple[List[str], List[str]]:
    """Check for missing and inconsistent columns against expected schema.

    Args:
        df (pd.DataFrame): Input dataframe to validate.
        expected_columns (List[str]): List of expected column names from configuration.

    Returns:
        Tuple[List[str], List[str]]:
            - List of missing columns that should be present
            - List of inconsistent (extra) columns not in expected schema
    """

    dataset_columns = set(df.columns)
    expected_columns_set = set(expected_columns)

    missing_columns = list(expected_columns_set - dataset_columns)
    inconsistent_columns = list(dataset_columns - expected_columns_set)

    return missing_columns, inconsistent_columns


def check_data_types(
    df: pd.DataFrame, column_types: Dict[str, str]
) -> Tuple[int, Dict[str, Dict[str, str]]]:
    """Check data type consistency across columns against expected types.

    Args:
        df (pd.DataFrame): Input dataframe to validate.
        column_types (Dict[str, str]): Expected column types from configuration.

    Returns:
        Tuple[int, Dict[str, Dict[str, str]]]:
            - Number of columns with type issues
            - Dictionary with type issue details for each problematic column
    """

    type_issues = {}

    # Set a generic number of categories for categorical columns
    number_of_expected_categories = 20

    for column, expected_type in column_types.items():
        if column not in df.columns:
            continue

        actual_dtype = str(df[column].dtype)
        issue = None

        # Map pandas dtypes to expected types
        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[column]):
                issue = "Not numeric"

        elif expected_type == "categorical":
            # For categorical, check if the number of unique values is reasonable
            unique_vals = df[column].dropna().nunique()

            if unique_vals > number_of_expected_categories:
                issue = "Not categorical. Many unique values."

        if issue:
            type_issues[column] = {
                "expected": expected_type,
                "actual": actual_dtype,
                "issue": issue,
            }

    return len(type_issues), type_issues


def check_duplicate_rows(df: pd.DataFrame) -> List[int]:
    """Check for duplicate rows in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to check for duplicate rows.

    Returns:
        List[int]: List of row indices that are duplicates (including first occurrence).
    """

    duplicate_indices = df[df.duplicated(keep=False)].index.tolist()

    return duplicate_indices


def statistical_analysis(
    synth_data: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on synthetic data.

    Args:
        synth_data (pd.DataFrame): Synthetic dataset to analyze.
        config (Dict[str, Any]): Configuration dictionary containing analysis parameters.

    Returns:
        Dict[str, Any]: Dictionary containing all statistical analysis results including
                       missing values, column consistency, data types, and duplicates.
    """

    # Init statistical results
    statistical_analysis_results = {}

    # Information
    statistical_analysis_results["information"] = config.get("information").get(
        "statistical_analysis"
    )

    # Missing values
    total_missing_values, rows_with_missing_indices, missing_counts_per_column = (
        check_missing_values(synth_data)
    )
    statistical_analysis_results["missing_values"] = {
        "total_missing_values": total_missing_values,
        "indices_with_missing_values": rows_with_missing_indices,
        "missing_values_per_column": missing_counts_per_column,
    }

    # Columns existence
    missing_columns, inconsistent_columns = check_column_existence(
        df=synth_data, expected_columns=config.get("expected_columns")
    )
    statistical_analysis_results["columns_consistency"] = {
        "missing_columns": missing_columns,
        "inconsistent_columns": inconsistent_columns,
    }

    # Data types
    num_issues, data_type_issues = check_data_types(
        df=synth_data, column_types=config.get("column_types")
    )
    statistical_analysis_results["data_types_consistency"] = {
        "number_inconsistent_data_types": num_issues,
        "data_types_issues": data_type_issues,
    }

    # Duplicate rows
    statistical_analysis_results["duplicate_rows_indices"] = check_duplicate_rows(
        synth_data
    )

    return statistical_analysis_results


def run_statistical_analysis(
    synth_path: Path,
    output_path: Path,
) -> None:
    """Run comprehensive statistical analysis on synthetic data for quality assessment.

    Args:
        synth_path (Path): Path to synthetic data CSV file.
        output_path (Path): Path where statistical analysis results will be saved.
    """

    # Load data
    synth_data = load_csv(synth_path)

    # Load config
    config = get_config()

    target = config.get("target_column")
    if target and (target in synth_data.columns):
        synth_data.drop(columns=[target], inplace=True)

    pass_checks(synth=synth_data, config=config)

    # Run statistical analysis
    statistical_analysis_results = statistical_analysis(
        synth_data=synth_data, config=config
    )

    # Store results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_json(data=statistical_analysis_results, path=output_path)

    print(f"Statistical Analysis Completed. Results saved to {output_path}")


def main() -> None:
    """Main entry point for statistical analysis of COPowereD synthetic data.

    Parses command line arguments and executes comprehensive statistical
    analysis including missing values, column consistency, data types,
    and duplicate detection.
    """

    parser = argparse.ArgumentParser(
        description="Run statistical analysis for COPowereD synthetic data"
    )
    parser.add_argument(
        "--synth_data", required=True, help="Path to tabular synthetic data CSV file"
    )

    parser.add_argument(
        "--output",
        default="output/statistical_analysis_results.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Execute statistical analysis
    run_statistical_analysis(
        synth_path=Path(args.synth_data),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
