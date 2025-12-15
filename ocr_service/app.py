# app.py
import os
import io
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import sqlite3

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient


# =========================================================
# CONFIG (ENV VARIABLES)
# =========================================================
AZURE_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_FORM_KEY")
MODEL_ID = os.getenv("AZURE_MODEL_ID", "handwritten_form_model_v2")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise RuntimeError(
        "Missing Azure credentials. Set AZURE_FORM_ENDPOINT and AZURE_FORM_KEY environment variables.\n"
        "Copy .env.example to .env and add your credentials."
    )


# =========================================================
# STORAGE
# =========================================================
BASE_DIR = "storage"
ORIG_DIR = f"{BASE_DIR}/originals"
SNIP_DIR = f"{BASE_DIR}/snippets"
ANNOT_DIR = f"{BASE_DIR}/annotated"

for d in [ORIG_DIR, SNIP_DIR, ANNOT_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================
DB_PATH = "corrections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            image_filename TEXT,
            field_name TEXT,
            bbox TEXT,
            original_text TEXT,
            corrected_text TEXT,
            snippet_path TEXT,
            extra_json TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# =========================================================
# APP + AZURE CLIENT
# =========================================================
app = FastAPI(title="Azure Document Intelligence OCR")

document_client = DocumentAnalysisClient(
    AZURE_ENDPOINT,
    AzureKeyCredential(AZURE_KEY)
)

# CORS for local file-based UI and other origins during prototyping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI at /ui
if os.path.isdir("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


# =========================================================
# SAFE VALUE CONVERTER  ✅ MOST IMPORTANT
# =========================================================
def safe_value(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [safe_value(x) for x in v]
    if isinstance(v, dict):
        return {k: safe_value(x) for k, x in v.items()}
    return str(v)


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(content: bytes) -> Image.Image:
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )
    return Image.fromarray(bin_img).convert("RGB")


# =========================================================
# AZURE ANALYSIS
# =========================================================
def analyze_with_azure(pil_image: Image.Image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)

    poller = document_client.begin_analyze_document(
        model_id=MODEL_ID,
        document=buf
    )
    return poller.result()


# =========================================================
# BBOX HELPERS
# =========================================================
def polygon_to_bbox(polygon):
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def crop_and_save(img: Image.Image, bbox: List[float], prefix: str):
    x1, y1, x2, y2 = map(int, bbox)
    crop = img.crop((x1, y1, x2, y2))
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    path = f"{SNIP_DIR}/{name}"
    crop.save(path)
    # return route path for UI consumption
    return f"snippets/{name}"


# =========================================================
# DRAW ANNOTATED IMAGE
# =========================================================
def draw_annotations(img: Image.Image, boxes: List[Dict]):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    for b in boxes:
        draw.rectangle(b["bbox"], outline="red", width=2)
        draw.text((b["bbox"][0], b["bbox"][1]), b["label"], fill="red", font=font)

    name = f"annot_{uuid.uuid4().hex[:8]}.jpg"
    path = f"{ANNOT_DIR}/{name}"
    out.save(path)
    # return route path so UI can fetch directly via server
    return f"annotated/{name}"


# =========================================================
# PARSE AZURE RESULT
# =========================================================
def parse_result(result, img: Image.Image):
    fields = {}
    boxes = []
    line_items = None
    invoice_total = None
    invoice_total_bbox = None

    if result.documents:
        for doc in result.documents:
            for name, field in doc.fields.items():
                # Skip Items and InvoiceTotal from top-level fields
                if name.lower() in ["items", "invoicetotal"]:
                    if name.lower() == "invoicetotal":
                        invoice_total = safe_value(field.value)
                        if getattr(field, "bounding_regions", None):
                            region = field.bounding_regions[0]
                            poly = getattr(region, "polygon", None)
                            if poly:
                                invoice_total_bbox = polygon_to_bbox(poly)
                    continue
                
                bbox = None
                if getattr(field, "bounding_regions", None):
                    region = field.bounding_regions[0]
                    poly = getattr(region, "polygon", None)
                    if poly:
                        bbox = polygon_to_bbox(poly)

                fields[name] = {
                    "value": safe_value(field.value),
                    "confidence": field.confidence,
                    "bbox": bbox
                }

                if bbox:
                    boxes.append({"label": name, "bbox": bbox})

            # Extract line items from Items field (array of dictionaries)
            items_field = doc.fields.get("Items") or doc.fields.get("items")
            if items_field and getattr(items_field, "value_type", None) == "list":
                items_array = items_field.value or []
                if items_array:
                    # Detect columns from first item and reorder
                    first_item = items_array[0]
                    if hasattr(first_item, "value") and isinstance(first_item.value, dict):
                        # Define column order: PurchaseOrderNo, Quantity, Description, Amount
                        desired_order = ["PurchaseOrderNo", "Quantity", "Description", "Amount"]
                        available_keys = list(first_item.value.keys())
                        # Use desired order, then append any extra columns
                        col_keys = [k for k in desired_order if k in available_keys]
                        col_keys += [k for k in available_keys if k not in col_keys]
                        columns = col_keys
                        
                        rows = []
                        for idx, item_field in enumerate(items_array):
                            if not hasattr(item_field, "value") or not isinstance(item_field.value, dict):
                                continue
                            
                            cells = []
                            for col_key in col_keys:
                                sub_field = item_field.value.get(col_key)
                                text = ""
                                bbox = None
                                
                                if sub_field:
                                    text = safe_value(getattr(sub_field, "value", "")) or getattr(sub_field, "content", "") or ""
                                    if getattr(sub_field, "bounding_regions", None):
                                        region = sub_field.bounding_regions[0]
                                        poly = getattr(region, "polygon", None)
                                        if poly:
                                            bbox = polygon_to_bbox(poly)
                                
                                cells.append({
                                    "col": col_key,
                                    "key": col_key,
                                    "text": text,
                                    "bbox": bbox
                                })
                            
                            rows.append({"row": idx, "cells": cells})
                        
                        # Add total row
                        if invoice_total is not None:
                            total_cells = []
                            for col_key in col_keys:
                                if col_key == "Amount":
                                    total_cells.append({
                                        "col": col_key,
                                        "key": col_key,
                                        "text": invoice_total,
                                        "bbox": invoice_total_bbox,
                                        "is_total": True
                                    })
                                elif col_key == "Description":
                                    total_cells.append({
                                        "col": col_key,
                                        "key": col_key,
                                        "text": "Total",
                                        "bbox": None,
                                        "is_total": True
                                    })
                                else:
                                    total_cells.append({
                                        "col": col_key,
                                        "key": col_key,
                                        "text": "",
                                        "bbox": None,
                                        "is_total": True
                                    })
                            rows.append({"row": "total", "cells": total_cells, "is_total": True})
                        
                        line_items = {"columns": columns, "rows": rows}

    annotated = draw_annotations(img, boxes)

    return {
        "fields": fields,
        "annotated_image": annotated,
        "line_items": line_items
    }


# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()

    orig_name = f"{ORIG_DIR}/orig_{uuid.uuid4().hex[:8]}.jpg"
    with open(orig_name, "wb") as f:
        f.write(content)

    # granular debug to pinpoint failure origin
    try:
        pre = preprocess_image(content)
    except Exception as e:
        print("PREPROCESS ERROR >>>", repr(e))
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = analyze_with_azure(pre)
    except Exception as e:
        print("AZURE ERROR >>>", repr(e))
        raise HTTPException(status_code=502, detail=str(e))

    try:
        parsed = parse_result(result, pre)
    except Exception as e:
        print("PARSE ERROR >>>", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

    parsed["original_image"] = orig_name
    parsed["timestamp"] = datetime.utcnow().isoformat()
    parsed["model_id"] = MODEL_ID

    return JSONResponse(parsed)

if not AZURE_ENDPOINT or not AZURE_KEY or not MODEL_ID:
    raise HTTPException(
        status_code=500,
        detail="Azure configuration missing. Check environment variables."
    )


# =========================================================
# SERVE ANNOTATED IMAGE
# =========================================================
@app.get("/annotated/{name}")
def get_annotated(name: str):
    path = f"{ANNOT_DIR}/{name}"
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path)
@app.get("/snippets/{name}")
def get_snippet(name: str):
    path = f"{SNIP_DIR}/{name}"
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path)



# =========================================================
# SAVE CORRECTIONS
# =========================================================
@app.post("/corrections")
def save_correction(payload: Dict[str, Any]):
    try:
        image_filename = payload.get("image_filename")
        field_name = payload.get("field_name")
        original_text = payload.get("original_text")
        corrected_text = payload.get("corrected_text")
        bbox = payload.get("bbox")  # [x1,y1,x2,y2]

        if not image_filename or not os.path.exists(image_filename):
            raise HTTPException(400, "image_filename missing or not found")

        snippet_path = None
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            try:
                with Image.open(image_filename) as im:
                    snippet_path = crop_and_save(im, bbox, prefix=field_name or "field")
            except Exception as e:
                # Non-fatal: still store correction even if snippet fails
                print("SNIPPET ERROR >>>", repr(e))

        row_id = uuid.uuid4().hex
        ts = datetime.utcnow().isoformat()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO corrections (
                id, timestamp, image_filename, field_name, bbox,
                original_text, corrected_text, snippet_path, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                ts,
                image_filename,
                field_name,
                json.dumps(bbox) if bbox is not None else None,
                original_text,
                corrected_text,
                snippet_path,
                json.dumps({k: v for k, v in payload.items() if k not in {
                    "image_filename", "field_name", "bbox", "original_text", "corrected_text"
                }})
            )
        )
        conn.commit()
        conn.close()

        return {"status": "ok", "id": row_id, "snippet_path": snippet_path}
    except HTTPException:
        raise
    except Exception as e:
        print("CORRECTION SAVE ERROR >>>", repr(e))
        raise HTTPException(500, str(e))


# =========================================================
# LIST CORRECTIONS
# =========================================================
@app.get("/corrections")
def list_corrections(limit: int = 50):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, image_filename, field_name, bbox,
                   original_text, corrected_text, snippet_path, extra_json
            FROM corrections
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cur.fetchall()
        conn.close()

        items = []
        for r in rows:
            bbox = None
            try:
                bbox = json.loads(r[4]) if r[4] else None
            except Exception:
                bbox = None
            extra = None
            try:
                extra = json.loads(r[8]) if r[8] else None
            except Exception:
                extra = None

            items.append({
                "id": r[0],
                "timestamp": r[1],
                "image_filename": r[2],
                "field_name": r[3],
                "bbox": bbox,
                "original_text": r[5],
                "corrected_text": r[6],
                "snippet_path": r[7],
                "extra_json": extra,
            })

        return {"items": items}
    except Exception as e:
        print("CORRECTION LIST ERROR >>>", repr(e))
        raise HTTPException(500, str(e))


# =========================================================
# SAVE DRAFT (BATCH)
# =========================================================
@app.post("/draft")
def save_draft(payload: Dict[str, Any]):
    try:
        draft_id = uuid.uuid4().hex
        ts = datetime.utcnow().isoformat()
        images_data = payload.get("images", [])

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Create drafts table if needed
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drafts (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                images_count INTEGER,
                images_json TEXT
            )
        """)

        cur.execute(
            "INSERT INTO drafts (id, timestamp, images_count, images_json) VALUES (?, ?, ?, ?)",
            (draft_id, ts, len(images_data), json.dumps(images_data))
        )
        conn.commit()
        conn.close()

        return {"status": "ok", "draft_id": draft_id, "timestamp": ts}
    except Exception as e:
        print("DRAFT SAVE ERROR >>>", repr(e))
        raise HTTPException(500, str(e))


# =========================================================
# EXPORT CSV (BATCH)
# =========================================================
@app.post("/export-csv")
def export_csv(payload: Dict[str, Any]):
    try:
        import csv
        from io import StringIO

        images = payload.get("images", [])
        if not images:
            raise HTTPException(400, "No images to export")

        output = StringIO()
        fieldnames = ["Image", "Field", "Value"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for img in images:
            filename = img.get("_filename", "unknown")
            
            # Write top-level fields
            fields = img.get("fields", {})
            for field_name, field_data in fields.items():
                value = field_data.get("value", "")
                writer.writerow({
                    "Image": filename,
                    "Field": field_name,
                    "Value": value
                })

            # Write line items
            line_items = img.get("line_items")
            if line_items and line_items.get("rows"):
                for row in line_items["rows"]:
                    if row.get("is_total"):
                        continue
                    for cell in row.get("cells", []):
                        col_key = cell.get("key", "")
                        text = cell.get("text", "")
                        writer.writerow({
                            "Image": filename,
                            "Field": f"Items.{col_key}",
                            "Value": text
                        })

        csv_content = output.getvalue()
        output.close()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=invoice_batch.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        print("CSV EXPORT ERROR >>>", repr(e))
        raise HTTPException(500, str(e))


# =========================================================
# VERCEL HANDLER
# =========================================================
# Vercel expects a handler function or app object
handler = app