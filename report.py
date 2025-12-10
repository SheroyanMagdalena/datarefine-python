import os
from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_pdf_report(
    output_path: str,
    df: pd.DataFrame,
    artifacts: dict,
    correlation_img_path: Optional[str] = None,
    rf_img_path: Optional[str] = None,
    xgb_img_path: Optional[str] = None,
    title: str = "DataRefine AutoML Report"
):
    """
    Generates a styled PDF report from artifacts and dataframe.

    Parameters
    ----------
    output_path : str
        Path to save the generated PDF.
    df : pd.DataFrame
        The final DataFrame after pipeline.
    artifacts : dict
        The full artifacts dictionary from pipeline stages.
    correlation_img_path : str, optional
        PNG path of correlation heatmap.
    rf_img_path : str, optional
        PNG path of RandomForest feature importances.
    xgb_img_path : str, optional
        PNG path of XGBoost feature importances.
    title : str
        Title of the report.

    Returns
    -------
    str
        Path to the saved PDF.
    """

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Title
    flow.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    flow.append(Spacer(1, 12))

    # --- Dataset Summary ---
    flow.append(Paragraph("<b>Dataset Overview</b>", styles["Heading2"]))
    flow.append(Paragraph(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", styles["Normal"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Columns:", styles["Heading3"]))
    for col in df.columns:
        flow.append(Paragraph(f"- {col}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    # --- Correlation Heatmap ---
    if correlation_img_path and os.path.exists(correlation_img_path):
        flow.append(Paragraph("<b>Correlation Matrix</b>", styles["Heading2"]))
        flow.append(Image(correlation_img_path, width=400, height=250))
        flow.append(Spacer(1, 12))

    # --- EDA Artifacts ---
    if "eda" in artifacts:
        flow.append(Paragraph("<b>Exploratory Data Analysis</b>", styles["Heading2"]))
        for step, content in artifacts["eda"].items():
            flow.append(Paragraph(f"<b>{step.replace('_', ' ').title()}</b>", styles["Heading3"]))
            if isinstance(content, dict):
                for key, value in content.items():
                    # show only scalar values or summaries
                    if isinstance(value, (int, float, str)):
                        flow.append(Paragraph(f"{key}: {value}", styles["Normal"]))
            flow.append(Spacer(1, 6))

    # --- Modeling Results ---
    if "modeling" in artifacts:
        flow.append(Paragraph("<b>Modeling Results</b>", styles["Heading2"]))

        for model_name, model_artifacts in artifacts["modeling"].items():
            flow.append(Paragraph(f"<b>{model_name.replace('_', ' ').title()}</b>", styles["Heading3"]))

            # Metrics
            metrics = model_artifacts.get("metrics")
            if isinstance(metrics, dict):
                data = [["Metric", "Value"]]
                for k, v in metrics.items():
                    data.append([k, round(v, 4) if isinstance(v, float) else v])
                t = Table(data, colWidths=[150, 200])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]))
                flow.append(t)
                flow.append(Spacer(1, 12))

            # Feature Importance Plot
            if "random_forest" in model_name and rf_img_path and os.path.exists(rf_img_path):
                flow.append(Image(rf_img_path, width=400, height=250))
            if "xgboost" in model_name and xgb_img_path and os.path.exists(xgb_img_path):
                flow.append(Image(xgb_img_path, width=400, height=250))

            flow.append(Spacer(1, 12))

    # --- Final Save ---
    doc.build(flow)
    return output_path
