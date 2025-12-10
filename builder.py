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
from templates.eda.numerical_vs_categorical import (
    get_numeric_and_categorical_columns,
)
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
    Executes a series of processing/modeling steps against an uploaded dataset
    and returns the final DataFrame + artifacts.

    Supports TWO config shapes:

    1) NEW (used by Next.js UI now):
       {
         "steps": [
           {"module": "cleaning", "function": "handle_missing"},
           {"module": "eda", "function": "head"},
           {"module": "encoding", "function": "onehot"},
           {"module": "scaling", "function": "standard"},
           {"module": "modeling", "function": "random_forest"}
         ]
       }

    2) OLD (stage-based, still supported for backwards compatibility):
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

    # Helper: execute one step
    def _run_step(stage: str, step_name: str):
        nonlocal current_df, artifacts

        if stage not in REGISTRY:
            if verbose:
                print(f"Skipping unknown stage '{stage}'")
            return

        func = REGISTRY[stage].get(step_name)
        if func is None:
            if verbose:
                print(f"Unknown step '{step_name}' in stage '{stage}', skipping.")
            return

        if verbose:
            print(f"- Executing: {stage}.{step_name}")

        # Ensure stage bucket exists
        artifacts.setdefault(stage, {})

        # --- Call the underlying function ---
        if stage == "modeling":
            if target is None:
                raise ValueError("Target column must be provided for modeling stage.")
            result = func(current_df, target=target)  # modeling gets target
        elif stage in ("encoding", "scaling"):
             # encoding & scaling may need target to skip it
            result = func(current_df, target=target)
        else:
            result = func(current_df)

        # --- Interpret result ---
        if isinstance(result, pd.DataFrame):
            # Bare DataFrame -> pipeline continues with updated df
            current_df = result
            return

        if isinstance(result, dict):
            # df + other outputs
            if "df" in result and isinstance(result["df"], pd.DataFrame):
                current_df = result["df"]

            # Store all non-df parts as artifacts
            artifacts[stage][step_name] = {
                k: v for k, v in result.items() if k != "df"
            }
        else:
            # Any other return type -> store raw as artifact
            artifacts[stage][step_name] = result

    # 3) Decide which config shape we're dealing with
    if isinstance(user_json, dict) and "steps" in user_json and isinstance(
        user_json["steps"], (list, tuple)
    ):
        # NEW STYLE: flat list of {"module", "function"}
        if verbose:
            print("=== Using NEW 'steps' config format ===")

        for idx, step in enumerate(user_json["steps"]):
            if not isinstance(step, dict):
                if verbose:
                    print(f"Skipping non-dict step at index {idx}: {step!r}")
                continue

            stage = step.get("module")
            step_name = step.get("function")

            if not isinstance(stage, str) or not isinstance(step_name, str):
                if verbose:
                    print(
                        f"Skipping malformed step at index {idx}: "
                        f"module={stage!r}, function={step_name!r}"
                    )
                continue

            if verbose:
                print(f"\n=== Running step {idx + 1}: {stage}.{step_name} ===")

            _run_step(stage, step_name)

    else:
        # OLD STYLE: { "eda": [...], "cleaning": [...], ... }
        if verbose:
            print("=== Using OLD stage-based config format ===")

        for stage, steps in user_json.items():
            if verbose:
                print(f"\n=== Running stage: {stage} ===")

            if stage not in REGISTRY:
                if verbose:
                    print(f"Skipping unknown stage '{stage}'")
                continue

            if not isinstance(steps, (list, tuple)):
                if verbose:
                    print(
                        f"Skipping stage '{stage}': steps is not a list/tuple "
                        f"(got {type(steps).__name__})"
                    )
                continue

            for step_name in steps:
                if not isinstance(step_name, str):
                    if verbose:
                        print(
                            f"Skipping non-string step in stage '{stage}': "
                            f"{step_name!r}"
                        )
                    continue

                _run_step(stage, step_name)

    # 4) Final output
    return {
        "df": current_df,
        "artifacts": artifacts,
    }


# Example usage when running this file directly
if __name__ == "__main__":
    # Old format example (still works)
    pipeline_json_old = {
        "eda": ["head", "summary_stats", "correlation"],
        "cleaning": ["handle_missing"],
        "encoding": ["onehot"],
        "scaling": ["standard"],
        "modeling": ["random_forest"],
    }

    # New format example (matches Next.js UI now)
    pipeline_json_new = {
        "steps": [
            {"module": "cleaning", "function": "handle_missing"},
            {"module": "eda", "function": "head"},
            {"module": "eda", "function": "summary_stats"},
            {"module": "eda", "function": "correlation"},
            {"module": "encoding", "function": "onehot"},
            {"module": "scaling", "function": "standard"},
            {"module": "modeling", "function": "random_forest"},
        ]
    }

    # Pick which example to test
    pipeline_json = pipeline_json_new

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