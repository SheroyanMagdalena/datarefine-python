# builder.py

import pandas as pd

# ----- Import all approved functions -----
from templates.cleaning.detect_outliers import detect_outliers
from templates.cleaning.feature_engineering import add_feature_engineering
from templates.cleaning.handle_missing import handle_missing
from templates.cleaning.normalize_formats import normalize_date_formats
from templates.cleaning.remove_duplicates import remove_duplicates

from templates.eda.convert_objects import convert_objects_to_numeric
from templates.eda.correlation import plot_correlation_matrix
from templates.eda.head import get_head
from templates.eda.numerical_vs_categorical import get_numeric_and_categorical_columns
from templates.eda.shape import get_shape
from templates.eda.summary_stats import get_summary_stats

from templates.encoding.onehot import one_hot_encode
from templates.encoding.ordinal import ordinal_encode

from templates.modeling.logistic_regression import train_logistic_regression
from templates.modeling.random_forest import train_random_forest
from templates.modeling.xgboost import train_xgboost

from templates.scaling.minmax import apply_minmax_scaler
from templates.scaling.standard import apply_standard_scaler


def build_pipeline(user_json, file_path, file_type="csv", target=None, verbose=True):
    """
    Executes a series of processing/modeling steps against an uploaded dataset,
    based on user_json, and returns the final DataFrame + artifacts.

    user_json example:
    {
        "eda": ["head", "summary_stats"],
        "cleaning": ["handle_missing"],
        "encoding": ["onehot"],
        "scaling": ["standard"],
        "modeling": ["random_forest"]
    }
    """

    # 1) Load dataset
    if file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type in ["xls", "xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # This will be updated as we apply transformations
    current_df = df

    # Collect non-DF outputs: metrics, summaries, plots, etc.
    artifacts: dict[str, dict] = {}

    # 2) Function registry by stage and template name
    # Keys in user_json must match these keys.
    REGISTRY = {
        "cleaning": {
            "detect_outliers": detect_outliers,                # df -> dict (no df change)
            "feature_engineering": add_feature_engineering,    # df -> df
            "handle_missing": handle_missing,                  # df -> df
            "normalize_formats": normalize_date_formats,       # df -> df
            "remove_duplicates": remove_duplicates,            # df -> dict(df + meta)
        },
        "eda": {
            "convert_objects": convert_objects_to_numeric,             # df -> dict(df + meta)
            "correlation": plot_correlation_matrix,                    # df -> dict
            "head": get_head,                                          # df -> dict
            "numerical_vs_categorical": get_numeric_and_categorical_columns,  # df -> dict
            "shape": get_shape,                                        # df -> dict
            "summary_stats": get_summary_stats,                        # df -> dict
        },
        "encoding": {
            "onehot": one_hot_encode,          # df -> dict(df + meta)
            "ordinal": ordinal_encode,         # df -> dict(df + meta)
        },
        "scaling": {
            "minmax": apply_minmax_scaler,     # df -> dict(df + meta)
            "standard": apply_standard_scaler, # df -> dict(df + meta)
        },
        "modeling": {
            "logistic_regression": train_logistic_regression,  # (df, target) -> dict
            "random_forest": train_random_forest,              # (df, target) -> dict
            "xgboost": train_xgboost,                          # (df, target) -> dict
        },
    }

    # 3) Execute stages in order
    for stage, steps in user_json.items():
        if verbose:
            print(f"\n=== Running stage: {stage} ===")

        if stage not in REGISTRY:
            if verbose:
                print(f"Skipping unknown stage '{stage}'")
            continue

        if not isinstance(steps, (list, tuple)):
            if verbose:
                print(f"Skipping stage '{stage}': steps is not a list/tuple")
            continue

        artifacts.setdefault(stage, {})

        for step_name in steps:
            if not isinstance(step_name, str):
                if verbose:
                    print(f"Skipping non-string step in stage '{stage}': {step_name!r}")
                continue

            func = REGISTRY[stage].get(step_name)
            if func is None:
                if verbose:
                    print(f"Unknown step '{step_name}' in stage '{stage}', skipping.")
                continue

            if verbose:
                print(f"- Executing: {stage}.{step_name}")

            # ---- Call function depending on type of step ----
            # Most functions take df; modeling also needs target.
            if stage == "modeling":
                if target is None:
                    raise ValueError("Target column must be provided for modeling stage.")
                result = func(current_df, target=target)
            else:
                result = func(current_df)

            # ---- Interpret the result ----
            # 1) If a function returns a bare DataFrame -> update current_df
            if isinstance(result, pd.DataFrame):
                current_df = result
                continue

            # 2) If it returns a dict, we may have:
            #    - "df" key (updated df)
            #    - other keys (metrics, summaries, plots, masks, etc.)
            if isinstance(result, dict):
                if "df" in result and isinstance(result["df"], pd.DataFrame):
                    current_df = result["df"]

                # Store everything else as artifacts
                artifacts[stage][step_name] = {
                    k: v for k, v in result.items() if k != "df"
                }

            else:
                # 3) Any other return type: store directly as an artifact
                artifacts[stage][step_name] = result

    # 4) Final output
    return {
        "df": current_df,
        "artifacts": artifacts,
    }


# Example usage when running this file directly
if __name__ == "__main__":
    pipeline_json = {
        "eda": ["head", "summary_stats", "correlation"],
        "cleaning": ["handle_missing"],
        "encoding": ["onehot"],
        "scaling": ["standard"],
        "modeling": ["random_forest"],
    }

    result = build_pipeline(
        user_json=pipeline_json,
        file_path="uploaded_dataset.csv",
        target="target",
        verbose=True,
    )

    final_df = result["df"]
    artifacts = result["artifacts"]

    print("\nFinal DF head:")
    print(final_df.head())

    print("\nArtifacts keys:")
    for stage, stuff in artifacts.items():
        print(stage, "->", list(stuff.keys()))