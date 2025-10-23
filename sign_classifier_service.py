# sign_classifier_service.py
# -*- coding: utf-8 -*-
import os
import time
import cv2
import base64
import tempfile
import numpy as np
from typing import List, Deque, Dict, Any
from collections import defaultdict, deque, Counter
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from ultralytics import YOLO

# ================= إعدادات عامة =================
MODEL_PATH = "best.pt"
DEFAULT_IMGSZ = 244
DEFAULT_INFER_MS = 200
STABILIZE_K = 5
CONF_THRESH = 0.55

# ================ Gemini (اختياري) ===================
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

GEMINI_MODEL = "gemini-2.0-flash-exp"

def _get_gemini_client(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def _gemini_sentence_from_top3(top3: List[tuple], api_key: str) -> str:
    """
    top3: list[(label:str, conf:float)]
    """
    if not GEMINI_AVAILABLE:
        return "(Gemini غير مثبت: pip install google-generativeai)"
    if not api_key:
        return "(Gemini غير مفعّل: لا يوجد API key)"
    try:
        model = _get_gemini_client(api_key, GEMINI_MODEL)
        labels = ", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in top3])
        prompt = (
            "أنت مساعد لوصف إشارات اليد بإيجاز ودقة.\n"
            f"هذه أفضل ثلاث نتائج (label مع أعلى ثقة شوهدت أثناء الجلسة): {labels}\n"
            "كوّن جملة عربية قصيرة وواضحة تربط هذه الإشارات في عبارة واحدة مفهومة للمراقِب."
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        msg = str(e)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            return "❌ مفتاح Gemini غير صالح/غير مُفعّل لمشروع عليه Generative Language API."
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return "❌ حساب/مشروع غير مخوّل للوصول إلى Generative Language API."
        return f"(Gemini Error) {msg}"

# تحميل النموذج مرة واحدة (يناسب بيئات الإنتاج مع عدد عمّال محدود)
model = YOLO(MODEL_PATH)

# ======================= حالة الجلسة =======================
class _VideoProcessor:
    def __init__(self):
        self.recent: Deque[tuple[str, float]] = deque(maxlen=STABILIZE_K)
        self.stable_label: str = ""
        self.stable_conf: float = 0.0
        self.last_pred_ts: float = 0.0

    def process_frame(self, frame_b64: str, imgsz: int = DEFAULT_IMGSZ, infer_every_ms: int = DEFAULT_INFER_MS) -> Dict[str, Any]:
        try:
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "فشل في فك تشفير الإطار"}

            now = time.time()
            if (now - self.last_pred_ts) * 1000.0 >= infer_every_ms:
                self.last_pred_ts = now

                results = model.predict(frame, imgsz=imgsz, verbose=False)
                r0 = results[0]

                # نتوقع نموذج تصنيف: r0.probs يجب أن تكون موجودة
                if getattr(r0, "probs", None) is None:
                    return {"error": "النموذج الحالي لا يُعيد احتمالات التصنيف (probs). تأكد أن model=تصنيف."}

                label_idx = int(r0.probs.top1)
                label = r0.names[label_idx]
                conf = float(r0.probs.top1conf)

                record = {"ts": now, "label": label, "conf": conf}
                _SESSION_DATA["records"].append(record)
                self.recent.append((label, conf))

                labels_only = [x[0] for x in self.recent]
                majority = Counter(labels_only).most_common(1)[0][0]
                max_conf_for_majority = max(
                    [c for (l, c) in self.recent if l == majority], default=0.0
                )

                if max_conf_for_majority >= CONF_THRESH:
                    self.stable_label, self.stable_conf = majority, max_conf_for_majority

                return {
                    "current_label": label,
                    "current_conf": conf,
                    "stable_label": self.stable_label,
                    "stable_conf": self.stable_conf,
                    "total_records": len(_SESSION_DATA["records"]),
                }

            return {
                "stable_label": self.stable_label,
                "stable_conf": self.stable_conf,
                "total_records": len(_SESSION_DATA["records"]),
            }

        except Exception as e:
            return {"error": f"خطأ في معالجة الإطار: {str(e)}"}

# حالة عامة بسيطة (جلسة واحدة). لو احتجت تعدد جلسات، أضف session_id.
_SESSION_DATA: Dict[str, Any] = {
    "recording": False,
    "records": [],  # {"ts": float, "label": str, "conf": float}
}

_processor = _VideoProcessor()

# ======================= نماذج الطلبات =======================
class ProcessFrameRequest(BaseModel):
    frame: str = Field(..., description="الإطار Base64 (JPEG/PNG)")
    imgsz: int = Field(DEFAULT_IMGSZ, description="مقاس الإدخال للنموذج")
    infer_every_ms: int = Field(DEFAULT_INFER_MS, description="الفاصل بين الاستدلالات بالملّي ثانية")

class TopItem(BaseModel):
    label: str
    confidence: float

class GeminiSummaryRequest(BaseModel):
    top3: List[TopItem]
    api_key: str

# ======================= الراوتر =======================
router = APIRouter(tags=["sign"])

@router.post("/start_session")
def start_session():
    _SESSION_DATA["recording"] = True
    _SESSION_DATA["records"] = []
    _processor.recent.clear()
    _processor.stable_label = ""
    _processor.stable_conf = 0.0
    _processor.last_pred_ts = 0.0
    return {"success": True, "message": "تم بدء الجلسة بنجاح"}

@router.post("/process_frame")
def process_frame(req: ProcessFrameRequest):
    if not _SESSION_DATA.get("recording"):
        raise HTTPException(status_code=400, detail="لا توجد جلسة نشطة")
    result = _processor.process_frame(req.frame, req.imgsz, req.infer_every_ms)
    return result

@router.post("/stop_session")
def stop_session():
    if not _SESSION_DATA.get("recording"):
        raise HTTPException(status_code=400, detail="لا توجد جلسة نشطة")

    _SESSION_DATA["recording"] = False

    if not _SESSION_DATA["records"]:
        return {"success": True, "message": "لا توجد نتائج كافية", "results": []}

    max_conf_per_label: Dict[str, float] = defaultdict(float)
    count_per_label: Dict[str, int] = defaultdict(int)

    for rec in _SESSION_DATA["records"]:
        lbl, cf = rec["label"], float(rec["conf"])
        count_per_label[lbl] += 1
        if cf > max_conf_per_label[lbl]:
            max_conf_per_label[lbl] = cf

    sorted_labels = sorted(max_conf_per_label.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_labels[:3]

    results = []
    for i, (lbl, conf) in enumerate(sorted_labels, 1):
        results.append({
            "rank": i,
            "label": lbl,
            "max_confidence": float(conf),
            "count": count_per_label[lbl]
        })

    return {
        "success": True,
        "total_records": len(_SESSION_DATA["records"]),
        "results": results,
        "top3": [{"label": lbl, "confidence": float(conf)} for lbl, conf in top3]
    }

@router.post("/get_gemini_summary")
def get_gemini_summary(req: GeminiSummaryRequest):
    top3_pairs = [(it.label, float(it.confidence)) for it in req.top3]
    summary = _gemini_sentence_from_top3(top3_pairs, req.api_key)
    return {"success": True, "summary": summary}

@router.post("/upload_image")
def upload_image(image: UploadFile = File(...), imgsz: int = Form(DEFAULT_IMGSZ)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="لم يتم اختيار ملف")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1] or ".jpg")
        try:
            content = image.file.read()
            tmp.write(content)
            tmp.flush()

            results = model.predict(source=tmp.name, imgsz=imgsz, verbose=False)

            predictions = []
            for r in results:
                if getattr(r, "probs", None) is None:
                    raise HTTPException(
                        status_code=500,
                        detail="النموذج الحالي لا يُعيد احتمالات تصنيف (probs). تأكد أنّك تستخدم نموذج تصنيف."
                    )
                label_idx = int(r.probs.top1)
                label = r.names[label_idx]
                conf = float(r.probs.top1conf)
                predictions.append({"label": label, "confidence": conf})

            return {"success": True, "predictions": predictions}
        finally:
            try:
                tmp.close()
                os.unlink(tmp.name)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gemini_available": GEMINI_AVAILABLE
    }
