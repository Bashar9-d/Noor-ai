# sign_text_service.py
# -*- coding: utf-8 -*-
import os, re, random
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from rapidfuzz import process, fuzz

load_dotenv()
CSV_INPUT = (os.getenv("SIGN_CSV_PATH", "") or "sign_videos.csv").strip()

router = APIRouter(tags=["signtext"])

SIGN_VIDEOS: Dict[str, List[str]] = {}
WORDS: List[str] = []
AR_MAPPING: Dict[str, str] = {}
_AR_DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')

_AR_NORMALIZED_TO_AR: Dict[str, str] = {}
_LAST_LOAD_INFO: Dict[str, Optional[str]] = {"path": None, "error": None, "rows": None}

def ar_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = _AR_DIAC.sub('', s)
    s = (s.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ٱ','ا')
           .replace('ؤ','و').replace('ئ','ي').replace('ى','ي').replace('ة','ه').replace('ـ',''))
    return s

def _best_fuzzy(term: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    match, score, _ = process.extractOne(term, candidates, scorer=fuzz.ratio)
    return match if score >= 60 else None

def _load_sign_videos(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV غير موجود: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError("CSV فارغ")

    df.columns = df.columns.str.lower().str.strip()
    required = {"gloss", "gloss_ar", "url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"أعمدة مفقودة في CSV: {missing}")

    videos: Dict[str, List[str]] = {}
    mapping: Dict[str, str] = {}

    for _, row in df.iterrows():
        g = str(row["gloss"]).strip().lower()
        ga = str(row["gloss_ar"]).strip().lower()
        u = str(row["url"]).strip()
        if not g or not u:
            continue
        videos.setdefault(g, []).append(u)
        if ga:
            mapping[ga] = g

    words = list(videos.keys())
    normalized_map: Dict[str, str] = {ar_normalize(ar): ar for ar in mapping.keys()}
    return videos, words, mapping, normalized_map, len(df)

def _assign_loaded(videos, words, mapping, normalized_map, rows, path):
    global SIGN_VIDEOS, WORDS, AR_MAPPING, _AR_NORMALIZED_TO_AR, _LAST_LOAD_INFO
    SIGN_VIDEOS = videos
    WORDS = words
    AR_MAPPING = mapping
    _AR_NORMALIZED_TO_AR = normalized_map
    _LAST_LOAD_INFO = {"path": path, "error": None, "rows": str(rows)}

def _fail_load(err: Exception, path: str):
    global SIGN_VIDEOS, WORDS, AR_MAPPING, _AR_NORMALIZED_TO_AR, _LAST_LOAD_INFO
    SIGN_VIDEOS, WORDS, AR_MAPPING, _AR_NORMALIZED_TO_AR = {}, [], {}, {}
    _LAST_LOAD_INFO = {"path": path, "error": str(err), "rows": None}

def initial_load():
    try:
        v, w, m, nmap, rows = _load_sign_videos(CSV_INPUT)
        _assign_loaded(v, w, m, nmap, rows, CSV_INPUT)
    except Exception as e:
        _fail_load(e, CSV_INPUT)

def spell_correct_from_csv(text: str) -> str:
    tokens = [t.lower().strip() for t in re.split(r"[,\s]+", text or "") if t]
    corrected_tokens: List[str] = []

    for t in tokens:
        if t in WORDS:
            corrected_tokens.append(t)
            continue
        if t in AR_MAPPING:
            corrected_tokens.append(AR_MAPPING[t])
            continue

        fuzzy_en = _best_fuzzy(t, WORDS)
        if fuzzy_en:
            corrected_tokens.append(fuzzy_en)
            continue

        norm_t = ar_normalize(t)
        ar_norm_keys = list(_AR_NORMALIZED_TO_AR.keys())
        fuzzy_norm = _best_fuzzy(norm_t, ar_norm_keys)
        if fuzzy_norm:
            ar_original = _AR_NORMALIZED_TO_AR[fuzzy_norm]
            corrected_tokens.append(AR_MAPPING[ar_original])
        else:
            corrected_tokens.append(t)

    return " ".join(corrected_tokens)

def normalize_sentence_to_glosses(text: str) -> List[str]:
    tokens = [t.lower().strip() for t in re.split(r"[,\s]+", text or "") if t]
    glosses: List[str] = []
    for t in tokens:
        if t in WORDS:
            glosses.append(t)
        elif t in AR_MAPPING:
            glosses.append(AR_MAPPING[t])
    return glosses

def build_playlist(glosses: List[str]) -> List[Dict[str, str]]:
    return [{"label": g, "url": random.choice(SIGN_VIDEOS[g])}
            for g in glosses if g in SIGN_VIDEOS and SIGN_VIDEOS[g]]

class InputText(BaseModel):
    text: str

class ReloadBody(BaseModel):
    csv_path: Optional[str] = None

@router.get("/health")
def health():
    return {
        "status": "ok" if WORDS else "degraded",
        "csv_path": _LAST_LOAD_INFO["path"],
        "rows": _LAST_LOAD_INFO["rows"],
        "error": _LAST_LOAD_INFO["error"],
        "counts": {
            "glosses_en": len(WORDS),
            "glosses_ar": len(AR_MAPPING),
            "videos": sum(len(v) for v in SIGN_VIDEOS.values()),
        },
    }

@router.post("/reload")
def reload_csv(body: ReloadBody):
    path = (body.csv_path or CSV_INPUT).strip()
    try:
        v, w, m, nmap, rows = _load_sign_videos(path)
        _assign_loaded(v, w, m, nmap, rows, path)
        return {"success": True, "csv_path": path, "rows": rows, "glosses": len(w)}
    except Exception as e:
        _fail_load(e, path)
        raise HTTPException(status_code=500, detail=f"فشل تحميل CSV: {e}")

@router.post("/process")
def process_text(data: InputText):
    if not WORDS:
        raise HTTPException(status_code=503, detail=f"البيانات غير محمّلة. آخر محاولة: path={_LAST_LOAD_INFO['path']} error={_LAST_LOAD_INFO['error']}")
    corrected = spell_correct_from_csv(data.text)
    glosses = normalize_sentence_to_glosses(corrected)
    playlist = build_playlist(glosses)
    return {"input": data.text, "corrected": corrected, "glosses": glosses, "playlist": playlist}
