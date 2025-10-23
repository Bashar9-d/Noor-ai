# -*- coding: utf-8 -*-
# services/vision_service.py
import os
import time
import json
import base64
import traceback
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

# ---------- تحميل إعدادات/مفاتيح من object.json (إن وُجد) ----------
PROJECT_DIR = Path(__file__).resolve().parent.parent  # up to project/
OBJECT_JSON_PATH = PROJECT_DIR / "object.json"

OBJECT_JSON: Dict[str, Any] = {}
if OBJECT_JSON_PATH.exists():
    try:
        with open(OBJECT_JSON_PATH, "r", encoding="utf-8") as f:
            OBJECT_JSON = json.load(f) or {}
    except Exception as e:
        print("[object.json] parse failed:", e)

def get_from_object(*keys, default: str = "") -> str:
    for k in keys:
        if isinstance(OBJECT_JSON, dict) and k in OBJECT_JSON and isinstance(OBJECT_JSON[k], str):
            val = OBJECT_JSON[k].strip()
            if val:
                return val
    return default

# ---------- (اختياري) Gemini ----------
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or get_from_object("GEMINI_API_KEY", "gemini_api_key", default="")
try:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if (USE_GEMINI and GEMINI_API_KEY) else None
except Exception:
    gemini_client = None

# ---------- Google Cloud TTS ----------
from google.cloud import texttospeech
from google.oauth2 import service_account

VOICE_DEFAULT = os.getenv("TTS_VOICE", "ar-XA-Standard-B")
TTS_LANG = os.getenv("TTS_LANG", "ar-XA")
TTS_RATE = float(os.getenv("TTS_RATE", "1.05"))
TTS_PITCH = float(os.getenv("TTS_PITCH", "0.0"))

def build_tts_client() -> texttospeech.TextToSpeechClient:
    try:
        if OBJECT_JSON and "type" in OBJECT_JSON and "private_key" in OBJECT_JSON:
            creds = service_account.Credentials.from_service_account_info(OBJECT_JSON)
            return texttospeech.TextToSpeechClient(credentials=creds)
        if "google_service_account" in OBJECT_JSON and isinstance(OBJECT_JSON["google_service_account"], dict):
            creds = service_account.Credentials.from_service_account_info(OBJECT_JSON["google_service_account"])
            return texttospeech.TextToSpeechClient(credentials=creds)
    except Exception as e:
        print("[TTS credentials] from object.json failed:", e)
        traceback.print_exc()
    return texttospeech.TextToSpeechClient()

# ========= خرائط التسمية =========
AR_NAME = {
    "person": "شخص", "bicycle": "دراجة", "car": "سيارة", "motorcycle": "دراجة نارية",
    "bus": "حافلة", "truck": "شاحنة", "bench": "مقعد", "chair": "كرسي", "couch": "كنبة",
    "dining table": "طاولة", "bottle": "زجاجة", "cup": "كوب", "fork": "شوكة", "knife": "سكين",
    "spoon": "ملعقة", "bowl": "وعاء", "banana": "موزة", "apple": "تفاحة", "sandwich": "شطيرة",
    "orange": "برتقالة", "broccoli": "بروكلي", "carrot": "جزرة", "pizza": "بيتزا",
    "donut": "دونات", "cake": "كيك", "potted plant": "نبتة", "tv": "شاشة",
    "laptop": "حاسوب محمول", "mouse": "فأرة", "remote": "ريموت", "keyboard": "لوحة مفاتيح",
    "cell phone": "هاتف", "book": "كتاب", "clock": "ساعة", "vase": "مزهرية",
    "scissors": "مقص", "toothbrush": "فرشاة أسنان",
}

def normalize_ar(s: str) -> str:
    s = (s or "").strip().lower()
    rep = {"أ":"ا","إ":"ا","آ":"ا","ى":"ي","ة":"ه","ؤ":"و","ئ":"ي","ـ":"",
           "ً":"","ٌ":"","ٍ":"","َ":"","ُ":"","ِ":"","ّ":""}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s

# ========= أدوات هندسية =========
def area_of(xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)

def pos_label_2d(cx, cy, W, H):
    col = "على اليسار" if cx < W/3 else ("في الوسط" if cx < 2*W/3 else "على اليمين")
    row = "في الأعلى" if cy < H/3 else ("في الوسط" if cy < 2*H/3 else "في الأسفل")
    return row, col

def humanize_pos(cx, cy, W, H):
    row, col = pos_label_2d(cx, cy, W, H)
    if row == "في الوسط" and col == "في الوسط": return "في الوسط"
    if row == "في الوسط": return col
    if col == "في الوسط": return row
    return f"{row}، {col}"

def distance_label(area_ratio=None, depth_val=None):
    if depth_val is not None:
        if depth_val <= 0.33: return "قريب"
        if depth_val <= 0.66: return "متوسط"
        return "بعيد"
    if area_ratio is not None:
        if area_ratio >= 0.10: return "قريب"
        elif area_ratio >= 0.03: return "متوسط"
        else: return "بعيد"
    return "متوسط"

# ========= حالة المشهد =========
import threading
class SceneState:
    def __init__(self):
        self.lock = threading.Lock()
        self.items: List[Dict[str, Any]] = []
        self.frame_size: Tuple[int, int] = (0, 0)
        self.last_update: float = 0.0
        self.model_names: Dict[int, str] = {}

    def update(self, items, frame_size, model_names):
        with self.lock:
            self.items = items
            self.frame_size = frame_size
            self.last_update = time.time()
            self.model_names = model_names

    def snapshot(self):
        with self.lock:
            return {
                "items": [dict(it) for it in self.items],
                "frame_size": tuple(self.frame_size),
                "last_update": self.last_update,
                "model_names": dict(self.model_names),
            }

SCENE = SceneState()

# ========= YOLO + (اختياري) MiDaS =========
class DepthEstimator:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ok = False
        try:
            self.model_type = "DPT_Large"
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device).eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
            self.ok = True
        except Exception as e:
            print("[DEPTH] init failed:", e)
            traceback.print_exc()
            self.ok = False

    def infer(self, frame_bgr):
        if not self.ok:
            return None
        with torch.no_grad():
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            im = self.transform(img).to(self.device)
            pred = self.model(im)
            depth = pred.squeeze().cpu().numpy()
            d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
            d = 1.0 - d
            return d

DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_IMG_SZ = 768
YOLO_CONF = 0.35
YOLO_IOU  = 0.60

try:
    yolo = YOLO("yolov8n-seg.pt")
    yolo.fuse()
except Exception as e:
    print("[YOLO] load failed:", e)
    traceback.print_exc()
    raise

DEPTH_ENABLED = bool(int(os.getenv("DEPTH_ENABLED", "1")))
DEPTH = DepthEstimator() if DEPTH_ENABLED else None
if DEPTH is not None and not DEPTH.ok:
    print("[DEPTH] Disabled (failed to init).")
    DEPTH = None

# ========= توصيف المشهد =========
def ar_count_phrase(n: int, noun_singular: str, noun_dual: str, noun_plural: str) -> str:
    if n == 0: return f"لا يوجد {noun_plural}."
    if n == 1: return f"{n} {noun_singular}."
    if n == 2: return f"{noun_dual}."
    return f"{n} {noun_plural}."

def describe_scene(snapshot: Dict[str, Any]) -> str:
    items = snapshot.get("items", [])
    if not items:
        return "لا أرى شيئًا واضحًا."
    counts: Dict[str, int] = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    by_count = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    parts_count = []
    for cls, n in by_count:
        ar = AR_NAME.get(cls, cls)
        if cls == "person":
            parts_count.append(ar_count_phrase(n, "شخص", "شخصان", "أشخاص").rstrip("."))
        else:
            parts_count.append(f"{n} {ar}")
    count_text = "، ".join(parts_count) if parts_count else "لا شيء مميز"
    top_items = sorted(items, key=lambda d: d.get("area_ratio", 0), reverse=True)[:3]
    loc_bits = [f"{it['label_ar']} {it.get('pos_text','')}" for it in top_items]
    loc_text = "؛ ".join(loc_bits)
    nearest = max(items, key=lambda d: d.get("area_ratio", 0))
    nearest_text = f"{nearest['label_ar']} {nearest.get('pos_text','')}, {nearest.get('distance','متوسط')}"
    return f"أرى: {count_text}. الأقرب: {nearest_text}. أمكنة بارزة: {loc_text}."

def nearest_brief(snapshot: Dict[str, Any]) -> str:
    items = snapshot.get("items", [])
    if not items:
        return "لا أرى شيئًا واضحًا."
    best = max(items, key=lambda d: d.get("area_ratio", 0))
    name = best.get("label_ar") or AR_NAME.get(best.get("label", ""), "شيء")
    pos  = best.get("pos_text", "")
    dist = best.get("distance", "متوسط")
    return f"الأقرب: {name} {pos}، {dist}."

def find_by_class(snapshot: Dict[str, Any], coco_name: str) -> List[Dict[str, Any]]:
    return [it for it in snapshot["items"] if it["label"] == coco_name]

def nearest_of(snapshot: Dict[str, Any], target_coco: str):
    hits = find_by_class(snapshot, target_coco)
    return None if not hits else max(hits, key=lambda d: d.get("area_ratio", 0))

def relative_loc(a, b):
    ax, ay = a["center"]; bx, by = b["center"]
    dx, dy = ax - bx, ay - by
    horiz = "يمين" if dx > 15 else ("يسار" if dx < -15 else "وسط")
    vert  = "أسفل" if dy > 15 else ("أعلى" if dy < -15 else "وسط")
    return f"{horiz} / {vert}"

INTENT_PATTERNS = {
    "describe": [r"(شو|ما|ايش).*(قدامي|امامي)|اوصف|وصف|اشرح"],
    "count":    [r"^(كم|عدد)"],
    "exist":    [r"^(هل|في |فيه|يوجد|موجود)"],
    "where":    [r"(وين|اين)"],
    "nearest":  [r"(اقرب|الاقرب|قريب)"],
    "rel_loc":  [r"(بالنسبه|بالنسبة)\s*ل"]
}
def parse_query_ar(text: str) -> Dict[str, Any]:
    t = normalize_ar(text)
    intent = ""
    for name, pats in INTENT_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            intent = name; break
    return {"intent": intent, "target": "", "ref": ""}

def is_scene_query(text: str) -> bool:
    t = normalize_ar(text)
    if any(k in t for k in ["شو قدامي","شو امامي","ما امامي","ايش قدامي","اوصف","وصف","اشرح المشهد","كم","عدد","هل","في ","فيه","يوجد","موجود","وين","اين","اقرب","الاقرب","قدامي","امامي"]):
        return True
    parsed = parse_query_ar(text)
    return parsed["intent"] in {"describe","count","exist","where","nearest","rel_loc"}

def answer_locally(user_text: str, snapshot: Dict[str, Any]) -> str:
    if time.time() - snapshot.get("last_update", 0) > 2.0:
        return "الصورة قديمة؛ التقط لقطة أحدث."
    parsed = parse_query_ar(user_text)
    intent = parsed["intent"]
    if intent == "describe":
        return describe_scene(snapshot)
    if intent == "nearest":
        return nearest_brief(snapshot)
    if intent == "count":
        persons = find_by_class(snapshot, "person")
        return ar_count_phrase(len(persons), "شخص", "شخصان", "أشخاص")
    if intent == "exist":
        return nearest_brief(snapshot)
    if intent == "where":
        return nearest_brief(snapshot)
    if intent == "rel_loc":
        return "لا أراهما معاً."
    if is_scene_query(user_text):
        return describe_scene(snapshot)
    return ""

# ---------- TTS: توليد MP3 وإرجاع Base64 (آمن) ----------
def tts_b64_safe(
    text: str,
    voice_name: str = VOICE_DEFAULT,
    speaking_rate: float = TTS_RATE,
    pitch: float = TTS_PITCH,
    language_code: str = TTS_LANG,
) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        client = build_tts_client()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=float(speaking_rate),
            pitch=float(pitch),
        )
        resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return base64.b64encode(resp.audio_content).decode("utf-8")
    except Exception as e:
        print("[TTS] failed:", e)
        traceback.print_exc()
        return ""

def gemini_answer(user_text: str, snapshot: Dict[str, Any]) -> str:
    if not gemini_client:
        return "لم أفهم السؤال."
    context = {
        "notice": "بيانات كائنات مرصودة من YOLOv8-seg على صورة الكاميرا الآن.",
        "frame_size": snapshot.get("frame_size"),
        "last_update_age_sec": round(time.time() - snapshot.get("last_update", 0), 2),
        "items": snapshot.get("items", []),
    }
    prompt = (
        "أنت مساعد صوتي عربي. أجب بإيجاز شديد وبالعربية الفصحى.\n"
        "لو كان السؤال يسأل عن وجود شيء محدد (نمط: في/هل/يوجد ...؟)، فالإجابة يجب أن تكون كلمة واحدة: "
        "\"نعم.\" أو \"لا.\" فقط.\n\n"
        f"سياق JSON (استخدمه فقط إن كان حديثًا):\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"السؤال: {user_text}\n"
        "أعد الجواب بجملة واحدة على الأكثر."
    )
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        elif getattr(resp, "candidates", None):
            return resp.candidates[0].content.parts[0].text.strip()
        return "لم أفهم السؤال."
    except Exception:
        return "لم أفهم السؤال."

# =========================
# Router
# =========================
router = APIRouter(tags=["Vision Assistant"])

@router.get("/health")
def health():
    return {"status": "ok", "gemini": bool(gemini_client), "depth": bool(DEPTH is not None)}

@router.post("/analyze_frame")
async def analyze_frame(
    file: UploadFile = File(...),
    voice_name: str = Form(VOICE_DEFAULT),
    rate: float = Form(TTS_RATE),
    pitch: float = Form(TTS_PITCH),
):
    try:
        data = await file.read()
        npimg = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        H, W = frame.shape[:2]
        depth_map = None
        if DEPTH and getattr(DEPTH, "ok", False):
            try:
                depth_map = DEPTH.infer(frame)
            except Exception as e:
                print("[DEPTH] infer failed:", e)
                traceback.print_exc()
                depth_map = None

        res = yolo.predict(
            source=frame, imgsz=YOLO_IMG_SZ, conf=YOLO_CONF, iou=YOLO_IOU,
            device=DEVICE, half=(DEVICE==0), verbose=False
        )[0]

        items = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy()
            masks_xy = res.masks.xy if (res.masks is not None) else None

            for i in range(len(xyxy)):
                bx = xyxy[i]; conf = float(confs[i]); cl = int(cls_ids[i])
                x1, y1, x2, y2 = bx
                cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
                a_ratio = float(area_of(bx) / (W * H + 1e-9))
                center = [int(cx), int(cy)]

                if masks_xy is not None and i < len(masks_xy):
                    poly = masks_xy[i]
                    try:
                        cnt = np.array(poly, dtype=np.float32)
                        area_pixels = abs(cv2.contourArea(cnt))
                        if area_pixels > 0:
                            a_ratio = float(area_pixels / (W * H + 1e-9))
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx_m = M["m10"] / M["m00"]; cy_m = M["m01"] / M["m00"]
                                center = [int(cx_m), int(cy_m)]
                    except Exception as e:
                        print("[MASK] moment failed:", e)

                depth_val = None
                if depth_map is not None:
                    x1i, y1i = max(0,int(x1)), max(0,int(y1))
                    x2i, y2i = min(W-1,int(x2)), min(H-1,int(y2))
                    crop = depth_map[y1i:y2i, x1i:x2i]
                    if crop.size > 0:
                        depth_val = float(np.median(crop))  # 0..1 (0 أقرب)

                pos_text = humanize_pos(center[0], center[1], W, H)
                dist = distance_label(area_ratio=a_ratio, depth_val=depth_val)
                cls_name = yolo.model.names[cl]
                label_ar = AR_NAME.get(cls_name, cls_name)

                items.append({
                    "cls_id": cl,
                    "label": cls_name,
                    "label_ar": label_ar,
                    "conf": round(conf, 3),
                    "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "center": center,
                    "area_ratio": round(a_ratio, 4),
                    "pos_text": pos_text,
                    "distance": dist,
                    "depth_val": None if depth_val is None else round(depth_val, 3)
                })

        SCENE.update(items, (H, W), yolo.model.names)
        snap = SCENE.snapshot()
        description = describe_scene(snap)

        audio_b64 = tts_b64_safe(description, voice_name=voice_name, speaking_rate=rate, pitch=pitch)

        snap["description"] = description
        snap["audio_base64"] = audio_b64
        return JSONResponse(snap)

    except Exception as e:
        print("[/analyze_frame] ERROR:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/ask")
async def ask(
    query: str = Form(...),
    voice_name: str = Form(VOICE_DEFAULT),
    rate: float = Form(TTS_RATE),
    pitch: float = Form(TTS_PITCH),
):
    try:
        user_text = (query or "").strip()
        if not user_text:
            return JSONResponse({"reply": "", "audio_base64": ""})
        snap = SCENE.snapshot()

        local = answer_locally(user_text, snap)
        if local:
            reply = local
        else:
            reply = gemini_answer(user_text, snap) if not is_scene_query(user_text) else describe_scene(snap)

        audio_b64 = tts_b64_safe(reply, voice_name=voice_name, speaking_rate=rate, pitch=pitch)
        return JSONResponse({"reply": reply, "audio_base64": audio_b64})

    except Exception as e:
        print("[/ask] ERROR:", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
