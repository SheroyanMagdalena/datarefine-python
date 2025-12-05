# builder.py  (was builder_upload.py)
import pandas as pd
from importlib import import_module


def safe_display(obj):
    """
    Replacement for Jupyter's display() so templates won't crash
    when run inside FastAPI. In a notebook it will still use the
    real display(), otherwise it just prints to stdout.
    """
    try:
        from IPython.display import display as ipy_display  # type: ignore
        ipy_display(obj)
    except Exception:
        print(obj)


def build_pipeline(user_json, file_path, file_type="csv", target=None, verbose=True):
    """
    Executes a series of code templates (EDA, cleaning, encoding, scaling, modeling)
    against an uploaded dataset, based on user_json, and returns the final DataFrame.

    user_json example:
    {
        "eda": ["head", "summary_stats"],
        "cleaning": ["handle_missing"],
        "encoding": ["onehot"],
        "scaling": ["standard"],
        "modeling": ["random_forest"]
    }
    """

    # 1) Load user dataset
    if file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type in ["xls", "xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # 2) Local variables exposed to templates
    #    Templates can assume df, target, pd, display, print are available.
    local_vars = {
        "df": df,
        "target": target,
        "pd": pd,
        "display": safe_display,  # <- so templates can call display(...)
        "print": print,           # <- explicitly expose print too
    }

    # 3) Execute templates in the order defined by user_json
    for stage, templates in user_json.items():
        if verbose:
            print(f"\n=== Running stage: {stage} ===")

        # be defensive: if someone sends booleans or non-lists in future
        if not isinstance(templates, (list, tuple)):
            if verbose:
                print(f"Skipping stage '{stage}' because templates is not a list/tuple")
            continue

        for tpl_name in templates:
            if not isinstance(tpl_name, str):
                if verbose:
                    print(f"Skipping template {tpl_name!r} in stage '{stage}' (not a string)")
                continue

            module_path = f"templates.{stage}.{tpl_name}"
            if verbose:
                print(f"Executing template: {tpl_name} (module: {module_path})")

            module = import_module(module_path)
            code_var = getattr(module, tpl_name.upper())

            # exec the template code in a sandboxed globals={}, shared locals=local_vars
            exec(code_var, {}, local_vars)

    # 4) Return final df (templates may have modified it in-place)
    return local_vars["df"]


# Example usage when running this file directly (for local testing)
if __name__ == "__main__":
    pipeline_json = {
        "eda": ["head", "summary_stats"],
        "cleaning": ["handle_missing"],
        "encoding": ["onehot"],
        "scaling": ["standard"],
        "modeling": ["random_forest"],
    }

    df_processed = build_pipeline(
        user_json=pipeline_json,
        file_path="uploaded_dataset.csv",
        target="target",
        verbose=True,
    )
    print(df_processed.head())
