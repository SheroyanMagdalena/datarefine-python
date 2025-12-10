import os
from typing import Optional, Dict, Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from xml.sax.saxutils import escape


def generate_pdf_report(
    output_path: str,
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    correlation_img_path: Optional[str] = None,
    rf_img_path: Optional[str] = None,
    xgb_img_path: Optional[str] = None,
    title: str = "DataRefine AutoML Report",
) -> str:
    """
    Generates a styled multi-section PDF report from the final DataFrame and pipeline artifacts.

    Sections:
    1) Cover / Title page
    2) Dataset Overview
    3) EDA Highlights (incl. correlation heatmap if provided)
    4) Modeling Results (metrics tables + feature importance plots)
    5) Pipeline / Artifacts Summary
    """

    # ---------- Setup document & styles ----------
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title=title,
    )

    styles = getSampleStyleSheet()

    # Custom styles (inspired by your reference)
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#2E86AB"),
            alignment=TA_CENTER,
            spaceAfter=24,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#A23B72"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#2E86AB"),
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontSize=12,
            textColor=colors.HexColor("#333333"),
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyText10",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#777777"),
            spaceAfter=4,
        )
    )

    flow = []

    # ============================================================
    # 1. COVER / TITLE "PAGE"
    # ============================================================
    flow.append(Spacer(1, 1.5 * inch))
    flow.append(Paragraph(escape(title), styles["ReportTitle"]))
    flow.append(
        Paragraph(
            "Automated Data Cleaning, EDA & Modeling Report",
            styles["ReportSubtitle"],
        )
    )
    flow.append(Spacer(1, 0.5 * inch))

    # Basic dataset metadata
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    meta_data = [
        ["Rows", str(n_rows)],
        ["Columns", str(n_cols)],
        ["Numeric Columns", str(len(numeric_cols))],
        ["Categorical Columns", str(len(cat_cols))],
    ]

    meta_table = Table(meta_data, colWidths=[2.5 * inch, 2.0 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    flow.append(meta_table)
    flow.append(Spacer(1, 0.7 * inch))
    flow.append(
        Paragraph(
            "<i>Generated automatically by DataRefine backend.</i>",
            styles["BodyText10"],
        )
    )

    # Soft page break to separate cover
    flow.append(PageBreak())

    # ============================================================
    # 2. DATASET OVERVIEW
    # ============================================================
    flow.append(Paragraph("1. Dataset Overview", styles["SectionHeading"]))

    overview_text = (
        f"This section summarizes the final dataset after the pipeline execution. "
        f"The processed dataset contains <b>{n_rows}</b> rows and <b>{n_cols}</b> columns."
    )
    flow.append(Paragraph(overview_text, styles["BodyText10"]))
    flow.append(Spacer(1, 6))

    # Show first N columns with dtype, missing, unique
    max_cols_preview = 10
    summary_rows = [
        ["Column", "Type", "Missing", "Unique"],
    ]

    for col in df.columns[:max_cols_preview]:
        col_series = df[col]
        summary_rows.append(
            [
                escape(str(col)),
                escape(str(col_series.dtype)),
                str(int(col_series.isna().sum())),
                str(int(col_series.nunique(dropna=True))),
            ]
        )

    if n_cols > max_cols_preview:
        summary_rows.append(["...", "...", "...", "..."])

    summary_table = Table(
        summary_rows,
        colWidths=[2.0 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch],
        repeatRows=1,
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86AB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    flow.append(summary_table)
    flow.append(Spacer(1, 12))

    # Optional: tiny head preview
    try:
        head_preview = df.head(5)
        flow.append(Paragraph("Sample Rows (first 5)", styles["SubHeading"]))
        head_data = [list(map(lambda c: escape(str(c)), head_preview.columns))]
        for _, row in head_preview.iterrows():
            head_data.append([escape(str(v)) for v in row.to_list()])

        head_table = Table(head_data, repeatRows=1)
        head_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ]
            )
        )
        flow.append(head_table)
        flow.append(Spacer(1, 18))
    except Exception:
        # In case there are weird types we can't tabulate
        flow.append(
            Paragraph(
                "Sample rows could not be rendered due to data types.",
                styles["SmallNote"],
            )
        )
        flow.append(Spacer(1, 12))

    # ============================================================
    # 3. EDA HIGHLIGHTS
    # ============================================================
    flow.append(Paragraph("2. Exploratory Data Analysis", styles["SectionHeading"]))

    eda_artifacts = artifacts.get("eda", {})

    if eda_artifacts:
        flow.append(
            Paragraph(
                "This section summarizes the main EDA steps and key derived statistics.",
                styles["BodyText10"],
            )
        )
        flow.append(Spacer(1, 6))

        # List which EDA steps ran
        flow.append(Paragraph("Executed EDA Steps:", styles["SubHeading"]))
        for step_name in eda_artifacts.keys():
            flow.append(
                Paragraph(
                    f"• {escape(step_name.replace('_', ' ').title())}",
                    styles["BodyText10"],
                )
            )
        flow.append(Spacer(1, 10))

        # Correlation heatmap image
        if correlation_img_path and os.path.exists(correlation_img_path):
            flow.append(Paragraph("Correlation Matrix Heatmap", styles["SubHeading"]))
            flow.append(
                Image(
                    correlation_img_path,
                    width=5.5 * inch,
                    height=3.5 * inch,
                    kind="proportional",
                )
            )
            flow.append(
                Paragraph(
                    "Figure: Numeric feature correlations.", styles["SmallNote"]
                )
            )
            flow.append(Spacer(1, 18))

        # If summary_stats present in artifacts["eda"]["summary_stats"], show a tiny subset
        summary_stats = eda_artifacts.get("summary_stats")
        if isinstance(summary_stats, dict):
            # Try to extract a compact stats table if it looks like a dict of dicts
            # (You can adjust this based on your summary_stats structure)
            flow.append(Paragraph("Statistical Summary (excerpt)", styles["SubHeading"]))
            # Best effort: show as "key: value" list
            shown = 0
            for k, v in summary_stats.items():
                if shown >= 12:
                    break
                flow.append(
                    Paragraph(
                        f"<b>{escape(str(k))}</b>: {escape(str(v))}",
                        styles["BodyText10"],
                    )
                )
                shown += 1
            flow.append(Spacer(1, 12))
    else:
        flow.append(
            Paragraph("No EDA artifacts were captured for this run.", styles["BodyText10"])
        )
        flow.append(Spacer(1, 12))

    # ============================================================
    # 4. MODELING RESULTS
    # ============================================================
    flow.append(Paragraph("3. Modeling Results", styles["SectionHeading"]))

    modeling_artifacts = artifacts.get("modeling", {})

    if modeling_artifacts:
        for model_name, model_art in modeling_artifacts.items():
            disp_name = model_name.replace("_", " ").title()
            flow.append(Paragraph(disp_name, styles["SubHeading"]))

            # Metrics table
            metrics = model_art.get("metrics")
            if isinstance(metrics, dict) and metrics:
                m_data = [["Metric", "Value"]]
                for mk, mv in metrics.items():
                    if isinstance(mv, float):
                        mv_disp = f"{mv:.4f}"
                    else:
                        mv_disp = str(mv)
                    m_data.append([escape(str(mk)), mv_disp])

                m_table = Table(m_data, colWidths=[2.0 * inch, 2.0 * inch])
                m_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#A23B72")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                        ]
                    )
                )
                flow.append(m_table)
                flow.append(Spacer(1, 8))

            # Classification report (if present)
            report_text = model_art.get("report")
            if isinstance(report_text, str) and report_text.strip():
                flow.append(
                    Paragraph("Classification Report (text)", styles["SmallNote"])
                )
                # Preserve newlines via <br/>
                safe_rep = escape(report_text).replace("\n", "<br/>")
                flow.append(Paragraph(f"<font size='7'>{safe_rep}</font>", styles["BodyText10"]))
                flow.append(Spacer(1, 10))

            # Feature importance image
            # Priority: explicit rf_img_path/xgb_img_path; fallback to model_art["feature_importance_plot"]
            feat_plot_path = model_art.get("feature_importance_plot")

            if (
                "random_forest" in model_name
                and rf_img_path
                and os.path.exists(rf_img_path)
            ):
                feat_plot_path = rf_img_path
            if "xgboost" in model_name and xgb_img_path and os.path.exists(xgb_img_path):
                feat_plot_path = xgb_img_path

            if feat_plot_path and os.path.exists(feat_plot_path):
                flow.append(
                    Paragraph("Feature Importance", styles["SmallNote"])
                )
                flow.append(
                    Image(
                        feat_plot_path,
                        width=5.5 * inch,
                        height=3.5 * inch,
                        kind="proportional",
                    )
                )
                flow.append(Spacer(1, 16))

        flow.append(Spacer(1, 12))
    else:
        flow.append(
            Paragraph("No modeling results were produced in this run.", styles["BodyText10"])
        )
        flow.append(Spacer(1, 12))

    # ============================================================
    # 5. PIPELINE / ARTIFACTS SUMMARY
    # ============================================================
    flow.append(Paragraph("4. Pipeline & Artifacts Summary", styles["SectionHeading"]))

    stages = list(artifacts.keys())
    if stages:
        flow.append(
            Paragraph(
                "Overview of stages that produced artifacts in this run:",
                styles["BodyText10"],
            )
        )
        flow.append(Spacer(1, 6))

        for stage in stages:
            flow.append(
                Paragraph(
                    f"<b>{escape(stage.title())}</b>", styles["BodyText10"]
                )
            )
            step_names = list(artifacts.get(stage, {}).keys())
            if step_names:
                for sn in step_names:
                    flow.append(
                        Paragraph(
                            f"• {escape(sn.replace('_', ' ').title())}",
                            styles["BodyText10"],
                        )
                    )
            else:
                flow.append(Paragraph("• (no sub-artifacts)", styles["BodyText10"]))
            flow.append(Spacer(1, 4))

    else:
        flow.append(
            Paragraph("No artifacts were recorded for this pipeline run.", styles["BodyText10"])
        )

    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(
            "<i>End of report.</i>",
            styles["SmallNote"],
        )
    )

    # ---------- Build PDF ----------
    doc.build(flow)
    return output_path
