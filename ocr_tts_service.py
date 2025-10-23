# -*- coding: utf-8 -*-
# services/ocr_tts_service.py
import os, io, re, base64
from typing import Optional, Tuple

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image, ExifTags
from dotenv import load_dotenv

# Google Cloud APIs
from google.cloud import vision
from google.cloud import texttospeech

# =========================
# Config
# =========================
load_dotenv()  # reads GOOGLE_APPLICATION_CREDENTIALS if in .env
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()
router = APIRouter(tags=["OCR/TTS"])

# =========================
# Helpers (نفس منطقك الأصلي)
# =========================
def _orientation_tag():
    for k, v in ExifTags.TAGS.items():
        if v == 'Orientation':
            return k
    return None

def fix_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        tag = _orientation_tag()
        exif = img._getexif() if hasattr(img, "_getexif") else None
        if not tag or not exif:
            return img
        orientation = exif.get(tag, 1)
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def img_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def has_arabic(s: str) -> bool:
    return any(
        ('\u0600' <= ch <= '\u06FF') or ('\u0750' <= ch <= '\u077F') or ('\u08A0' <= ch <= '\u08FF')
        for ch in s
    )

def map_lang_for_google_tts(ocr_lang_code: str, text: str) -> str:
    lc = (ocr_lang_code or "").lower()
    if lc.startswith("ar") or has_arabic(text):
        return "ar-XA"
    if lc.startswith("en"):
        return "en-US"
    return "en-US"

def vision_ocr(img_bytes: bytes, language_hints=("ar", "en")) -> Tuple[str, str]:
    image = vision.Image(content=img_bytes)
    response = vision_client.document_text_detection(
        image=image,
        image_context=vision.ImageContext(language_hints=list(language_hints)) if language_hints else None,
    )
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    lang_code = ""
    try:
        pages = response.full_text_annotation.pages
        if pages and pages[0].property.detected_languages:
            langs = sorted(pages[0].property.detected_languages, key=lambda x: x.confidence or 0, reverse=True)
            lang_code = langs[0].language_code if langs else ""
    except Exception:
        pass
    return (full_text.strip(), lang_code)

def google_tts(
    text: str,
    language_code: str,
    voice_name: str = "",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    audio_format: str = "MP3",
) -> bytes:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = (
        texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        if voice_name else
        texttospeech.VoiceSelectionParams(language_code=language_code)
    )
    encoding = getattr(texttospeech.AudioEncoding, audio_format, texttospeech.AudioEncoding.MP3)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=encoding,
        speaking_rate=float(speaking_rate),
        pitch=float(pitch),
    )
    resp = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    return resp.audio_content

# ---------- JOD detection (نفس منطقك) ----------
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
def normalize_ar_digits(s: str) -> str:
    return s.translate(_ARABIC_INDIC)

def parse_jod_denomination(text: str) -> Optional[int]:
    if not text:
        return None
    t = normalize_ar_digits(text.lower())
    money_words = r"(?:دينار|دنانير|jd|jod|dinar|dinars)"

    direct_patterns = {
        50: [rf"\b50\s*{money_words}\b", r"\bخمس(?:ون|ين)\s*(?:دينار|دنانير)\b", r"\bخمسون\b", r"\bخمسين\b",
             r"\bfifty\s*dinars?\b", r"\bfifty\b"],
        20: [rf"\b20\s*{money_words}\b", r"\bعشر(?:ون|ين)\s*(?:دينار|دنانير)\b", r"\bعشرون\b", r"\bعشرين\b",
             r"\btwenty\s*dinars?\b", r"\btwenty\b"],
        10: [rf"\b10\s*{money_words}\b", r"\bعشرة\s*(?:دينار|دنانير)\b", r"\bعشرة\b",
             r"\bten\s*dinars?\b", r"\bten\b"],
        5:  [rf"\b5\s*{money_words}\b", r"\bخمسة\s*(?:دينار|دنانير)\b", r"\bخمس\b",
             r"\bfive\s*dinars?\b", r"\bfive\b"],
        1:  [rf"(?<![0-9])\b1\s*{money_words}\b", r"\bدينار(?:\s*واحد)?\b", r"\bواحد(?:ة)?\s*دينار\b",
             r"\bone\s*dinar\b", r"\bone\b"]
    }
    for d, pats in direct_patterns.items():
        for pat in pats:
            if re.search(pat, t, flags=re.UNICODE):
                return d

    money_spans = [m.span() for m in re.finditer(money_words, t, flags=re.UNICODE)]
    if money_spans:
        num_tokens = [(m.group(), m.span()) for m in re.finditer(r"\b(?:50|20|10|5|1)\b", t)]
        for num, (s, e) in num_tokens:
            for (ms, me) in money_spans:
                if abs(s - ms) <= 14 or abs(e - me) <= 14:
                    return int(num)

    words_map = {
        50: [r"\bخمسون\b", r"\bخمسين\b", r"\bfifty\b"],
        20: [r"\bعشرون\b", r"\bعشرين\b", r"\btwenty\b"],
        10: [r"\bعشرة\b", r"\bten\b"],
        5:  [r"\bخمسة\b", r"\bخمس\b", r"\bfive\b"],
        1:  [r"\bواحد(?:ة)?\b", r"\bone\b"]
    }
    for d, pats in words_map.items():
        for pat in pats:
            if re.search(pat, t, flags=re.UNICODE):
                return d
    return None

def denomination_sentence_ar(denom: int) -> str:
    if denom == 1:  return "هذه ورقة نقدية أردنية فئة دينار واحد."
    if denom == 5:  return "هذه ورقة نقدية أردنية فئة خمسة دنانير."
    if denom == 10: return "هذه ورقة نقدية أردنية فئة عشرة دنانير."
    if denom == 20: return "هذه ورقة نقدية أردنية فئة عشرين دينارًا."
    if denom == 50: return "هذه ورقة نقدية أردنية فئة خمسين دينارًا."
    return "تعذّر تحديد الفئة."

# =========================
# API (تحت /ocr)
# =========================
@router.post("/tts")
async def ocr_tts(
    file: UploadFile = File(...),
    voice_name: str = Form(""),
    rate: float = Form(1.0),
    pitch: float = Form(0.0),
    lang_override: str = Form("", description="Optional: e.g., ar-XA or en-US"),
    audio_format: str = Form("MP3", description="MP3 | OGG_OPUS | LINEAR16"),
):
    try:
        pil_img = Image.open(io.BytesIO(await file.read()))
        pil_img = fix_exif_orientation(pil_img)
        img_bytes = img_to_bytes(pil_img)

        full_text, ocr_lang = vision_ocr(img_bytes, language_hints=("ar", "en"))
        if not full_text:
            return JSONResponse({"error": "No text found"}, status_code=400)

        denom = parse_jod_denomination(full_text)
        if denom:
            msg = denomination_sentence_ar(denom)
            ar_voice = voice_name if voice_name.strip().lower().startswith("ar-") else ""
            audio_bytes = google_tts(
                msg,
                language_code="ar-XA",
                voice_name=ar_voice,
                speaking_rate=rate,
                pitch=pitch,
                audio_format=audio_format,
            )
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {
                "detected": "currency",
                "denomination": denom,
                "text": msg,
                "audio_base64": audio_b64
            }

        lc = (lang_override.strip() or map_lang_for_google_tts(ocr_lang, full_text))
        audio_bytes = google_tts(
            full_text,
            language_code=lc,
            voice_name=voice_name,
            speaking_rate=rate,
            pitch=pitch,
            audio_format=audio_format,
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "detected": "text",
            "text": full_text,
            "lang": lc,
            "ocr_lang_code": ocr_lang,
            "audio_base64": audio_b64
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/health")
def health():
    return {"status": "ok", "service": "ocr-tts"}
