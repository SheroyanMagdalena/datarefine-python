import json
import tempfile
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from builder import build_pipeline

app = FastAPI()

# Optional: allow your Next.js origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/build-pipeline")
async def build_pipeline_endpoint(
    file: UploadFile = File(...),
    config: str = Form(...),        # JSON string from frontend
    file_type: str = Form("csv"),
    target: Optional[str] = Form(None),
):
    """
    1) Save uploaded file to temp
    2) Parse pipeline config JSON
    3) Call build_pipeline(...)
    4) Return a preview of resulting DataFrame
    """
    # Save uploaded file to a temp path
    contents = await file.read()
    suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    user_json = json.loads(config)

    df_processed = build_pipeline(
        user_json=user_json,
        file_path=tmp_path,
        file_type=file_type,
        target=target,
        verbose=True,
    )

    # You probably don’t want to send the whole DF; just a sample
    preview = df_processed.head(50).to_dict(orient="records")

    return {
        "preview": preview,
        "columns": list(df_processed.columns),
        # TODO later: maybe return profiling, charts, etc.
    }

