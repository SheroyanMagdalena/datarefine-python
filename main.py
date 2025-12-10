import os
import json
import tempfile
from typing import Optional
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from builder import build_pipeline
from report import generate_pdf_report  # adjust import if needed

app = FastAPI()

# Example CORS, adjust as you like
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/build-pipeline")
async def build_pipeline_endpoint(
    file: UploadFile = File(...),
    config: str = Form(...),
    file_type: str = Form("csv"),
    target: Optional[str] = Form(None),
):
    """
    1) Save uploaded file to temp
    2) Parse pipeline config JSON
    3) Call build_pipeline(...)
    4) Generate PDF report and return preview + artifacts
    """

    # --- Save uploaded file to temp ---
    contents = await file.read()
    suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    user_json = json.loads(config)

    # --- Build pipeline ---
    result = build_pipeline(
        user_json=user_json,
        file_path=tmp_path,
        file_type=file_type,
        target=target,
        verbose=True,
    )

    df_processed: pd.DataFrame = result["df"]
    artifacts = result["artifacts"]
    preview = df_processed.head(50).to_dict(orient="records")

    # --- Locate any plots for PDF ---
    correlation_img_path = (
        artifacts.get("eda", {})
        .get("correlation", {})
        .get("heatmap_plot")
    )
    rf_img_path = (
        artifacts.get("modeling", {})
        .get("random_forest", {})
        .get("feature_importance_plot")
    )
    xgb_img_path = (
        artifacts.get("modeling", {})
        .get("xgboost", {})
        .get("feature_importance_plot")
    )

    # --- Generate PDF report ---
    report_path = os.path.join(tempfile.gettempdir(), "report.pdf")
    generate_pdf_report(
        output_path=report_path,
        df=df_processed,
        artifacts=artifacts,
        correlation_img_path=correlation_img_path,
        rf_img_path=rf_img_path,
        xgb_img_path=xgb_img_path,
    )

    raw_response = {
        "preview": preview,
        "columns": list(df_processed.columns),
        "artifacts": artifacts,
        "report_path": report_path,
    }

    # --- Make everything JSON-safe ---
    safe_response = jsonable_encoder(
        raw_response,
        custom_encoder={
            # numpy scalars -> Python scalars
            np.generic: lambda x: x.item(),
            # numpy arrays -> lists
            np.ndarray: lambda x: x.tolist(),
        },
    )
    cleaned = replace_nans(safe_response)

    return JSONResponse(content=cleaned)

def replace_nans(obj):
    """
    Recursively walk a nested structure (dict/list/scalars)
    and replace NaN / inf floats with None so JSON is happy.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: replace_nans(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [replace_nans(v) for v in obj]

    # Leave everything else as-is (str, int, bool, None, etc.)
    return obj

@app.get("/download-report")
def download_report():
    """
    Serve the latest generated PDF report.
    """
    report_path = os.path.join(tempfile.gettempdir(), "report.pdf")

    if not os.path.exists(report_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Report not found."},
        )

    return FileResponse(
        path=report_path,
        filename="datarefine_report.pdf",
        media_type="application/pdf",
    )
