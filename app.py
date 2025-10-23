# main.py
# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocr_tts_service import router as ocr_router
from vision_service import router as vision_router
from sign_classifier_service import router as sign_router
from sign_text_service import router as signtext_router, initial_load as signtext_initial_load  # <-- استيراد المحمّل

app = FastAPI(title="Unified Vision + OCR→TTS Server", version="1.1")

# CORS عام
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # عدّلها لاحقاً لقيود أقوى
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# الراوترات تحت بادئات واضحة
app.include_router(ocr_router, prefix="/ocr")
app.include_router(vision_router, prefix="/vision")
app.include_router(sign_router, prefix="/sign")
app.include_router(signtext_router, prefix="/signtext")

# حمّل CSV عند بدء التطبيق (مهم)
@app.on_event("startup")
async def _startup():
    signtext_initial_load()

@app.get("/")
def root():
    return {
        "status": "ok",
        "routes": {
            # OCR
            "ocr_tts": "POST /ocr/tts",
            "ocr_health": "GET /ocr/health",

            # Vision
            "vision_analyze": "POST /vision/analyze_frame",
            "vision_ask": "POST /vision/ask",
            "vision_health": "GET /vision/health",

            # Sign classifier (YOLO)
            "sign_start": "POST /sign/start_session",
            "sign_frame": "POST /sign/process_frame",
            "sign_stop": "POST /sign/stop_session",
            "sign_gemini": "POST /sign/get_gemini_summary",
            "sign_upload": "POST /sign/upload_image",
            "sign_health": "GET /sign/health",

            # Sign text → playlist
            "signtext_process": "POST /signtext/process",
            "signtext_reload":  "POST /signtext/reload",
            "signtext_health":  "GET /signtext/health",

            "health": "GET /health",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "services": ["ocr", "vision", "sign", "signtext"]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
