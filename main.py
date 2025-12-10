from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Optional
import json
import tempfile
import os

import pandas as pd

from builder import build_pipeline
from report import generate_pdf_report

app = FastAPI()


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

    df_processed = result["df"]
    artifacts = result["artifacts"]
    preview = df_processed.head(50).to_dict(orient="records")

    # --- Locate any plots for PDF ---
    correlation_img_path = artifacts.get("eda", {}).get("correlation", {}).get("heatmap_plot")
    rf_img_path = artifacts.get("modeling", {}).get("random_forest", {}).get("feature_importance_plot")
    xgb_img_path = artifacts.get("modeling", {}).get("xgboost", {}).get("feature_importance_plot")

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

    return {
        "preview": preview,
        "columns": list(df_processed.columns),
        "artifacts": artifacts,
        "report_path": report_path,
    }

@app.get("/download-report")
def download_report():
    """
    Serve the latest generated PDF report.
    (You may want to make this user/session-specific in production.)
    """
    report_path = os.path.join(tempfile.gettempdir(), "report.pdf")

    if not os.path.exists(report_path):
        return {"error": "Report not found."}

    return FileResponse(
        path=report_path,
        filename="datarefine_report.pdf",
        media_type="application/pdf"
    )