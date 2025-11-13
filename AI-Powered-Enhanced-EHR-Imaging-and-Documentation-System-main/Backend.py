import os
import io
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Optional
from PIL import Image
import uvicorn # type: ignore

# optional transformers / keras imports are tried but not required to run
try:
    from transformers import pipeline
    hf_available = True
except Exception:
    hf_available = False

try:
    from tensorflow.keras.models import load_model # type: ignore
    tf_available = True
except Exception:
    tf_available = False

# -----------------------
# Config / Data
# -----------------------
ENHANCED_FOLDER = "Xray_enhanced"
PROCESSED_FOLDER = "Xray_processed"
os.makedirs(ENHANCED_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load ICD csv safely (if present)
ICD_CSV = "ICD10codes.csv"
df_icd = None
if os.path.exists(ICD_CSV):
    df_icd = pd.read_csv(ICD_CSV)
    df_icd.columns = df_icd.columns.str.strip()

# Try to load autoencoder (if you have one)
MODEL_FILE = "xray_enhancer_model.h5"
autoencoder = None
if tf_available and os.path.exists(MODEL_FILE):
    try:
        autoencoder = load_model(MODEL_FILE)
        print("✅ Autoencoder loaded.")
    except Exception as e:
        print("⚠️ Failed to load autoencoder:", e)

# Try to load a text generation model
generator = None
if hf_available:
    try:
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        print("✅ Hugging Face generator loaded.")
    except Exception as e:
        print("⚠️ HF pipeline not available:", e)
        generator = None

app = FastAPI(title="EHR Imaging Backend")

# -----------------------
# Utility functions
# -----------------------
def match_icd(diagnosis: str) -> str:
    if df_icd is None or not diagnosis:
        return "N/A"
    # find likely disease column and code column
    # attempt multiple heuristics because ICD CSVs come in many shapes
    code_col = df_icd.columns[0]
    disease_cols = [c for c in df_icd.columns if ("disease" in c.lower()) or ("description" in c.lower())]
    if not disease_cols:
        # fallback: pick longest string-like column
        disease_cols = [c for c in df_icd.columns if df_icd[c].dtype == object][:1]
    if not disease_cols:
        return "N/A"
    disease_col = disease_cols[0]
    diag = (diagnosis or "").lower()[:5]
    mask = df_icd[disease_col].astype(str).str.lower().str.contains(diag, na=False)
    match = df_icd[mask]
    if not match.empty:
        return str(match.iloc[0][code_col])
    return "N/A"

def enhance_image_bytes(img_bytes: bytes, save_name_prefix="enh") -> str:
    """Return path to enhanced image file. If autoencoder exists, use it; else use OpenCV filters."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bad image data")

    # Preprocess (resize + grayscale)
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # If autoencoder available, use it (expects normalized float32 input)
    if autoencoder is not None:
        inp = gray.astype("float32") / 255.0
        inp = np.expand_dims(inp, axis=(0, -1))
        try:
            enhanced = autoencoder.predict(inp)[0].squeeze()
            out_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        except Exception:
            # fallback to simple improvement
            out_uint8 = cv2.equalizeHist(gray)
    else:
        # Simple enhancement fallback (CLAHE + denoise)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        out_uint8 = clahe.apply(gray)
        out_uint8 = cv2.fastNlMeansDenoising(out_uint8, h=10)

    fname = f"{save_name_prefix}_{np.random.randint(1e6)}.png"
    path = os.path.join(ENHANCED_FOLDER, fname)
    cv2.imwrite(path, out_uint8)
    return path

def generate_summary(age, gender, symptom, diagnosis, lab, icd):
    prompt = f"Summarize clinical info briefly: Age {age}, Gender {gender}, Symptoms {symptom}. Lab: {lab}. Diagnosis: {diagnosis}. ICD-10: {icd}."
    if generator is not None:
        try:
            out = generator(
                prompt,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.3
            )[0]['generated_text']
            # Clean repetitive text like “Adults: Adults:”
            out = out.replace("Adults:", "").strip()
            return out
        except Exception:
            return f"Summary (fallback): {prompt}"
    else:
        # template fallback
        return f"Summary (template): {age}y {gender}. Symptoms: {symptom}. Diagnosis: {diagnosis}. Labs: {lab}. ICD-10: {icd}."


# -----------------------
# API Models
# -----------------------
class ProcessResponse(BaseModel):
    enhanced_image_path: str
    icd10: str
    clinical_summary: str

# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "EHR Imaging Backend ready."}

@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(...),
    age: Optional[str] = Form("N/A"),
    gender: Optional[str] = Form("N/A"),
    symptom: Optional[str] = Form("N/A"),
    diagnosis: Optional[str] = Form("N/A"),
    lab: Optional[str] = Form("N/A"),
):
    content = await file.read()
    try:
        enhanced_path = enhance_image_bytes(content)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Image processing failed: {e}"})
    icd = match_icd(diagnosis)
    summary = generate_summary(age, gender, symptom, diagnosis, lab, icd)
    return ProcessResponse(
        enhanced_image_path=enhanced_path,
        icd10=icd,
        clinical_summary=summary
    )

# -----------------------
# Run (if run directly)
# -----------------------
if __name__ == "__main__":
    uvicorn.run("Backend:app", host="127.0.0.1", port=8000, reload=True)
