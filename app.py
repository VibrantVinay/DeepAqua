import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import threading
import platform
import os
import json
import base64
import io
import wave
import zipfile
from datetime import datetime
from pathlib import Path
import pillow_avif  # noqa: F401  (registers AVIF support)
from PIL import Image
from ultralytics import YOLO

from utils.knowledge_engine import KnowledgeEngine
from utils.alert_system import AlertSystem

if platform.system() == "Windows":
    import winsound

# ==============================================================================
# CONFIG
# ==============================================================================
VLM_MODEL = os.getenv("DEEPAQUA_VLM_MODEL", "qwen/qwen3.8-27b")
VEHICLE_LABELS = ["SUBMARINE", "AUV", "ROV", "SHIP", "UUV", "GLIDER", "DIVER", "MINE", "ORDNANCE", "WEAPON", "HOSTILE"]
ALL_LABELS = ["Submarine", "Fish", "Ship", "AUV", "ROV", "Diver", "UUV", "Wreck", "Glider",
              "Plant", "Seagrass", "Moss", "Coral", "Reef", "Kelp", "Ecosystem",
              "Naval Mine", "Underwater Ordnance", "Suspicious Object"]

THEMES = {  # name: (primary, text-on-primary)
    "Ocean": ("#315cbd", "#ffffff"),
    "Violet": ("#8b5cf6", "#ffffff"),
    "Rose": ("#f43f5e", "#ffffff"),
    "Amber": ("#f59e0b", "#1c1003"),
    "Sky": ("#0ea5e9", "#03131f"),
}
MODES = {
    "Cream": dict(bg="#f4f0e8", fg="#111a27", card="#fcfaf5", muted="#e9e4da", mfg="#68717b", border="#d9d3c8"),
    "Dark": dict(bg="#08111d", fg="#f4f0e8", card="#0d1b2a", muted="#14263a", mfg="#a9b4bf", border="#26394e"),
}

st.set_page_config(page_title="DeepAqua Studio", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

for k, v in {"history": [], "last_alarm_time": 0.0, "last_alert_time": 0.0, "api_key": "", "mission_name": "Coastal survey", "operator_name": ""}.items():
    st.session_state.setdefault(k, v)

# ==============================================================================
# GROQ VLM (key comes from secrets / env / sidebar, never from source code)
# ==============================================================================
def _resolve_key():
    if st.session_state.api_key:
        return st.session_state.api_key
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

@st.cache_resource(show_spinner=False)
def _make_client(key):
    from groq import Groq
    return Groq(api_key=key)

def get_client():
    key = _resolve_key()
    if not key:
        return None
    try:
        return _make_client(key)
    except Exception:
        return None

def query_groq_vision(frame_bgr, prompt, max_tokens=150, temperature=0.1):
    client = get_client()
    if client is None:
        return "Offline"
    h, w = frame_bgr.shape[:2]
    if max(h, w) > 640:
        s = 640 / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return "VLM Error: encode failed"
    b64 = base64.b64encode(buf).decode()
    try:
        r = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        return "RATE LIMIT EXCEEDED" if ("429" in msg or "rate_limit" in msg.lower()) else f"VLM Error: {e}"

def _vlm_failed(text):
    return text in ("Offline", "RATE LIMIT EXCEEDED") or text.startswith("VLM Error")

def get_groq_supplementary_detections(frame_bgr):
    h, w = frame_bgr.shape[:2]
    prompt = f"""Analyze this underwater image. Detect supported marine life, plants, reefs, wrecks, debris, ships, submarines, AUV/ROV/UUV vehicles, divers, naval mines, and possible underwater ordnance. Image size is {w}x{h}.
Only call something a threat when visible evidence supports it; use "uncertain" in the label when the image is ambiguous.
Reply ONLY with a raw JSON array in this format:
[{{"label": "AUV", "confidence": 0.85, "bbox": [10, 20, 150, 200], "type": "threat"}}]
Use 'threat' for military vehicles, ships, divers, mines, or possible ordnance; use 'ecology' for marine life, plants, corals, wrecks, and debris. bbox is [x_min, y_min, x_max, y_max] in pixels.
If nothing is found return []. No markdown or backticks."""
    content = query_groq_vision(frame_bgr, prompt, max_tokens=600)
    if _vlm_failed(content):
        return [], []
    if "```" in content:
        content = content.split("```json")[-1].split("```")[0] if "```json" in content else content.split("```")[1]
    try:
        data = json.loads(content.strip())
        gt, ge = [], []
        for it in data:
            is_threat = str(it.get("type", "")).lower() == "threat"
            obj = {"label": it.get("label", "Unknown"), "confidence": float(it.get("confidence", 0.5)),
                   "bbox": it.get("bbox", [0, 0, 0, 0]), "severity": "CRITICAL" if is_threat else "INFO", "source": "VLM"}
            (gt if is_threat else ge).append(obj)
        return gt, ge
    except Exception:
        return [], []

def verify_label_with_groq(frame_bgr, bbox, original_label):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return original_label
    prompt = f"""Cropped underwater object. The first model guessed '{original_label}'.
Pick the MOST ACCURATE label from: {", ".join(ALL_LABELS)}. Reply with the single word only."""
    resp = query_groq_vision(frame_bgr[y1:y2, x1:x2], prompt, max_tokens=15).upper()
    if _vlm_failed(resp) or resp.startswith("VLM ERROR"):
        return original_label
    for cat in ALL_LABELS:
        if cat.upper() in resp:
            return cat
    return original_label

def run_deep_scan_groq(frame_bgr, bbox, is_threat):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
    kind = "underwater vehicle (e.g. 'Kilo-class submarine')" if is_threat else "biological entity, plant, or structure"
    return query_groq_vision(crop, f"Identify the exact model name or species of this {kind}. Return ONLY the specific name.", max_tokens=30)

def scene_summary(frame_bgr):
    return query_groq_vision(frame_bgr, "You are a subsea survey analyst. In 3 short sentences describe this scene: visibility, notable objects, and anything an operator should act on.", max_tokens=160, temperature=0.2)

# ==============================================================================
# IMAGE / DETECTION HELPERS
# ==============================================================================
def calculate_iou(a, b):
    xl, yt = max(a[0], b[0]), max(a[1], b[1])
    xr, yb = min(a[2], b[2]), min(a[3], b[3])
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0

def filter_duplicate_boxes(new, existing, thr=0.3):
    return [n for n in new if not any(calculate_iou(n["bbox"], e["bbox"]) > thr for e in existing)]

def enhance_underwater(bgr):
    """Gray-world white balance (recovers lost red) + CLAHE contrast on luminance."""
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / np.maximum(means, 1e-6)
    f = np.clip(f, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def hex_to_bgr(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))

def draw_verified_boxes(image, threats, ecology, thickness=2, show_conf=True):
    out = image.copy()
    for items, color in ((threats, (82, 90, 255)), (ecology, hex_to_bgr(THEMES[st.session_state.get("theme", "Ocean")][0]))):
        for t in items:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            txt = f"{t['label']} {t['confidence']:.0%}" if show_conf else t["label"]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytop = max(0, y1 - th - 8)
            cv2.rectangle(out, (x1, ytop), (x1 + tw + 8, ytop + th + 8), color, -1)
            cv2.putText(out, txt, (x1 + 4, ytop + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)
    return out

def split_by_vlm(items, frame, max_verify):
    threats, ecology = [], []
    for n, it in enumerate(items):
        if n < max_verify:
            it["label"] = verify_label_with_groq(frame, it["bbox"], it["label"])
        if any(term in it["label"].upper() for term in VEHICLE_LABELS):
            it["severity"] = "CRITICAL"; threats.append(it)
        else:
            it["severity"] = "INFO"; ecology.append(it)
    return threats, ecology

def apply_label_filter(items, allowed):
    return [i for i in items if i["label"].title() in allowed] if allowed else items

def to_dataframe(threats, ecology):
    rows = []
    for grp, items in (("Priority", threats), ("Environment", ecology)):
        for i in items:
            x1, y1, x2, y2 = [int(v) for v in i["bbox"]]
            rows.append({"Group": grp, "Label": i["label"], "Confidence": round(i["confidence"], 3),
                         "Source": i.get("source", "YOLO"), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows, columns=["Group", "Label", "Confidence", "Source", "x1", "y1", "x2", "y2"])

def png_bytes(bgr):
    return cv2.imencode(".png", bgr)[1].tobytes()

# ==============================================================================
# ALARMS
# ==============================================================================
def play_3_beeps():
    if platform.system() == "Windows":
        for _ in range(3):
            winsound.Beep(900, 400); time.sleep(0.15)

def make_beep_wav():
    sample_rate = 24000
    samples = np.zeros(int(sample_rate * 0.78), dtype=np.float32)
    for start in (0.0, 0.27, 0.54):
        left = int(start * sample_rate)
        right = min(len(samples), left + int(0.17 * sample_rate))
        t = np.arange(right - left, dtype=np.float32) / sample_rate
        fade = np.minimum(1.0, np.arange(len(t)) / 120) * np.minimum(1.0, np.arange(len(t))[::-1] / 120)
        samples[left:right] += 0.24 * fade * np.sin(2 * np.pi * 880 * t)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()

def trigger_threat_alarm(threats, active):
    if not active or not threats:
        return
    now = time.time()
    if now - st.session_state.last_alarm_time > 5.0:
        st.session_state.last_alarm_time = now
        threading.Thread(target=play_3_beeps, daemon=True).start()
        st.audio(make_beep_wav(), format="audio/wav", autoplay=True)
        st.toast(f"{len(threats)} priority contact(s) detected", icon="🚨")

def throttled_alert(alerter, threats, frame):
    if time.time() - st.session_state.last_alert_time > 5.0:
        st.session_state.last_alert_time = time.time()
        alerter.trigger(threats, frame)

# ==============================================================================
# THEME (shadcn design tokens)
# ==============================================================================
def inject_css(theme, mode):
    p, pfg = THEMES[theme]
    m = MODES[mode]
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
    :root{--bg:%bg%;--fg:%fg%;--card:%card%;--muted:%muted%;--mfg:%mfg%;--border:%border%;--primary:%p%;--pfg:%pfg%;
          --danger:#b6453c;--ok:#507b68;--metal:#b59b6b;--r:12px;--ring:%p%55;}
    html,body,.stApp{font-family:'Manrope',sans-serif!important;background:var(--bg);color:var(--fg);}
    .stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{color:var(--fg);}
    .stApp{background-image:radial-gradient(65% 35% at 100% 0%,color-mix(in srgb,var(--primary) 8%,transparent),transparent 72%),linear-gradient(145deg,var(--bg),color-mix(in srgb,var(--bg) 90%,var(--primary)));background-attachment:fixed;}
    header[data-testid="stHeader"]{background:transparent;}
    .block-container{padding-top:1.2rem;max-width:1440px;}
    section[data-testid="stSidebar"]{background:color-mix(in srgb,var(--card) 94%,var(--primary));border-right:1px solid var(--border);}
    hr{border-color:var(--border)!important;}
    /* cards */
    .card{background:color-mix(in srgb,var(--card) 92%,var(--primary));border:1px solid var(--border);border-radius:var(--r);padding:20px 22px;margin-bottom:14px;box-shadow:0 12px 34px rgba(8,17,29,.055);}
    .card h4{margin:0 0 4px;font-family:'Cormorant Garamond',serif;font-size:1.35rem;font-weight:600;letter-spacing:.01em;} .card .desc{color:var(--mfg);font-size:.84rem;margin:0;line-height:1.65;}
    .kpi{background:color-mix(in srgb,var(--card) 92%,var(--primary));border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;box-shadow:0 8px 24px rgba(8,17,29,.04);}
    .kpi .k{color:var(--mfg);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.11em;} .kpi .v{font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:600;letter-spacing:-.02em;line-height:1.15;}
    .kpi .s{color:var(--mfg);font-size:.74rem;}
    .kpi.danger{border-color:var(--danger);} .kpi.danger .v{color:var(--danger);}
    /* header */
    .appbar{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;margin:8px 0 24px;padding:26px 0 22px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);}.hero-copy{max-width:760px}.hero-kicker{display:flex;align-items:center;gap:10px;color:var(--metal);font-size:.68rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;margin-bottom:10px}.hero-kicker:before{content:"";display:block;width:28px;height:1px;background:var(--metal)}.hero-title{font-family:"Cormorant Garamond",serif;font-size:clamp(2.6rem,5vw,4.7rem);font-weight:500;letter-spacing:-.035em;line-height:.92}.hero-title em{color:var(--primary);font-style:normal}.hero-subtitle{max-width:700px;color:var(--mfg)!important;font-size:.88rem;line-height:1.7;margin-top:14px}.hero-status{display:flex;align-items:center;gap:7px;flex-wrap:wrap;justify-content:flex-end;max-width:430px}
    .brand{display:flex;align-items:center;gap:12px;padding-bottom:7px;} .logo{width:38px;height:38px;border-radius:50%;border:1px solid var(--metal);background:transparent;display:flex;align-items:center;justify-content:center;}
    .brand .n{font-family:'Cormorant Garamond',serif;font-weight:600;font-size:1.55rem;letter-spacing:.015em;} .brand .t{color:var(--mfg);font-size:.66rem;text-transform:uppercase;letter-spacing:.15em;}
    
    /* badges */
    .badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:5px 10px;font-size:.67rem;font-weight:700;letter-spacing:.035em;border:1px solid var(--border);background:var(--muted);color:var(--fg);margin:3px;}
    .badge.primary{background:transparent;color:var(--primary);border-color:color-mix(in srgb,var(--primary) 45%,var(--border));}
    .badge.danger{background:rgba(182,69,60,.1);color:var(--danger);border-color:rgba(182,69,60,.35);}
    .badge.ok{background:rgba(80,123,104,.1);color:var(--ok);border-color:rgba(80,123,104,.35);}
    .dot{width:7px;height:7px;border-radius:50%;background:currentColor;display:inline-block;}
    .det{display:flex;justify-content:space-between;align-items:center;gap:10px;background:var(--card);border:1px solid var(--border);border-left:2px solid var(--primary);border-radius:8px;padding:12px 14px;margin-bottom:8px;}
    .det.crit{border-left-color:var(--danger);} .det .l{font-weight:600;font-size:.9rem;} .det .c{color:var(--mfg);font-size:.78rem;}
    .bar{height:5px;background:var(--muted);border-radius:99px;overflow:hidden;width:90px;} .bar i{display:block;height:100%;background:var(--primary);}
    .det.crit .bar i{background:var(--danger);}
    /* streamlit widgets -> shadcn */
    .stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;padding:0;border-bottom:1px solid var(--border);width:100%;}
    .stTabs [data-baseweb="tab"]{height:42px;padding:0 14px;border-radius:0;color:var(--mfg);font-weight:600;font-size:.77rem;letter-spacing:.045em;background:transparent;}
    .stTabs [aria-selected="true"]{background:transparent!important;color:var(--primary)!important;box-shadow:inset 0 -2px 0 var(--primary)!important;}
    .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
    .stButton>button,.stDownloadButton>button{background:transparent;color:var(--fg);border:1px solid var(--border);border-radius:5px;font-weight:700;font-size:.76rem;letter-spacing:.04em;padding:8px 15px;transition:.18s;}
    .stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--primary);color:var(--fg);background:var(--muted);}
    .stButton>button:focus-visible,.stDownloadButton>button:focus-visible{box-shadow:0 0 0 3px var(--ring);outline:none;}
    .stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--primary);color:#fff;border-color:var(--primary);}
    [data-testid="stFileUploaderDropzone"]{background:color-mix(in srgb,var(--card) 92%,var(--primary));border:1px dashed color-mix(in srgb,var(--metal) 55%,var(--border));border-radius:var(--r);}
    [data-testid="stFileUploaderDropzone"]:hover{border-color:var(--primary);}[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploaderDropzone"] button *,button[aria-label="Upload"],button[aria-label="Upload"] *{background:#f4f0e8!important;color:#111a27!important;border:1px solid #777!important;border-radius:8px!important;font-family:Manrope,sans-serif!important;font-weight:700!important;opacity:1!important;}[data-testid="stFileUploaderDropzone"] button:hover,[data-testid="stFileUploaderDropzone"] button:hover *,button[aria-label="Upload"]:hover,button[aria-label="Upload"]:hover *{background:#315cbd!important;color:#fff!important;border-color:#315cbd!important;}{background:var(--muted)!important;color:var(--fg)!important;border:1px solid var(--border)!important;border-radius:8px!important;font-family:Manrope,sans-serif!important;font-weight:700!important;opacity:1!important;}[data-testid="stFileUploaderDropzone"] button:hover,[data-testid="stFileUploaderDropzone"] button:hover *{background:var(--primary)!important;color:#fff!important;border-color:var(--primary)!important;}
    .stTextInput input,.stNumberInput input,[data-baseweb="select"]>div{background:var(--card)!important;border-color:var(--border)!important;color:var(--fg)!important;border-radius:8px!important;}
    .stSlider [role="slider"]{background:var(--primary)!important;box-shadow:0 0 0 4px var(--ring)!important;}
    [data-testid="stExpander"]{border:1px solid var(--border);border-radius:var(--r);background:var(--card);}
    [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:var(--r);overflow:hidden;}
    [data-testid="stImage"] img{border-radius:var(--r);border:1px solid var(--border);box-shadow:0 14px 38px rgba(8,17,29,.08);}
    .side-h{color:var(--metal);font-size:.67rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;margin:18px 0 7px;}.stSlider [data-baseweb="slider"]>div:first-child{height:3px}@media(max-width:720px){.appbar{align-items:flex-start;padding:20px 0}.hero-status{justify-content:flex-start}.hero-title{font-size:3rem}.block-container{padding-left:1rem;padding-right:1rem}}
    </style>"""
    for k, v in {"%bg%": m["bg"], "%fg%": m["fg"], "%card%": m["card"], "%muted%": m["muted"], "%mfg%": m["mfg"],
                 "%border%": m["border"], "%p%": p, "%pfg%": pfg}.items():
        css = css.replace(k, v)
    st.markdown(css, unsafe_allow_html=True)

def kpi(label, value, sub="", danger=False):
    return f'<div class="kpi {"danger" if danger else ""}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

def det_row(item, crit, name):
    pct = item["confidence"] * 100
    return (f'<div class="det {"crit" if crit else ""}"><div><div class="l">{name}</div>'
            f'<div class="c">{item.get("source", "YOLO")} · {pct:.1f}% confidence</div></div>'
            f'<div class="bar"><i style="width:{pct:.0f}%"></i></div></div>')

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2.2" stroke-linecap="round" style="color:#fff"><path d="M2 8c2 2 4 3 6 3s4-2 6-2 4 1 6 1"/>'
                '<path d="M2 14c2 2 4 3 6 3s4-2 6-2 4 1 6 1"/></svg></div>'
                '<div><div class="n">DeepAqua</div><div class="t">Subsea Vision Studio</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-h">Appearance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    theme = c1.selectbox("Accent", list(THEMES), key="theme", label_visibility="collapsed")
    mode = c2.selectbox("Mode", list(MODES), key="mode", label_visibility="collapsed")

    st.markdown('<div class="side-h">Detection</div>', unsafe_allow_html=True)
    mission_profile = st.selectbox("Mission profile", ["Balanced", "Threat watch", "Ecology survey"], key="mission_profile")
    profile_threshold = {"Balanced": 0.50, "Threat watch": 0.35, "Ecology survey": 0.30}[mission_profile]
    conf_thresh = st.slider("Confidence threshold", 0.10, 1.00, profile_threshold, 0.05, key=f"confidence_{mission_profile}")
    iou_thresh = st.slider("IoU overlap threshold", 0.10, 1.00, 0.45, 0.05)
    label_filter = st.multiselect("Only show classes", ALL_LABELS, placeholder="All classes")
    enhance = st.toggle("Underwater enhancement", value=False, help="White-balance + contrast boost before detection.")

    st.markdown('<div class="side-h">VLM co-pilot</div>', unsafe_allow_html=True)
    use_groq_vision = st.toggle("Verify detections with VLM", value=True)
    max_verify = st.slider("Max verifications per frame", 1, 20, 8)
    with st.expander("Groq API key"):
        st.text_input("Key", type="password", key="api_key", placeholder="gsk_…",
                      help="Or set GROQ_API_KEY in .streamlit/secrets.toml / environment.")

    st.markdown('<div class="side-h">Overlay & alerts</div>', unsafe_allow_html=True)
    thickness = st.slider("Box thickness", 1, 6, 2)
    show_conf = st.toggle("Show confidence on boxes", value=True)
    audio_alarm_toggle = st.toggle("Audible threat alarm", value=True)
    alert_toggle = st.toggle("Save alert evidence", value=True)
    if st.button("Test alarm sound", use_container_width=True):
        st.audio(make_beep_wav(), format="audio/wav", autoplay=True)

inject_css(theme, mode)
GROQ_AVAILABLE = get_client() is not None

# ==============================================================================
# MODEL
# ==============================================================================
@st.cache_resource
def load_deepaqua_model():
    for p in (Path("models/best.pt"), Path("runs/detect/deepaqua_colab_run-2/weights/best.pt")):
        if p.exists():
            return YOLO(str(p)), p.name
    return YOLO("yolo11n.pt"), "yolo11n.pt (fallback)"

try:
    model, weights_name = load_deepaqua_model()
    engine, alerter = KnowledgeEngine(), AlertSystem()
except Exception as e:
    st.error(f"Couldn't load the detection model: {e}. Check that models/best.pt exists.")
    st.stop()

# ==============================================================================
# HEADER
# ==============================================================================
vlm_badge = '<span class="badge ok"><span class="dot"></span>VLM online</span>' if GROQ_AVAILABLE else \
            '<span class="badge danger"><span class="dot"></span>VLM offline · add API key</span>'
st.markdown(f"""
<div class="appbar">
  <div class="hero-copy"><div class="hero-kicker">DeepAqua · Subsea intelligence</div>
  <div class="hero-title">The subsea frontier,<br><em>understood.</em></div><div class="hero-subtitle">A precision workspace for marine reconnaissance. YOLO identifies trained underwater classes; VLM analysis adds context. Review uncertain contacts before acting.</div></div>
  <div class="hero-status"><span class="badge primary">{weights_name}</span>{vlm_badge}<span class="badge">Confidence ≥ {conf_thresh:.2f}</span></div>
</div>""", unsafe_allow_html=True)

tab_image, tab_video, tab_batch, tab_session = st.tabs(["Image recon", "Uploaded video", "Batch recon", "Session & report"])


def log_history(source, threats, ecology, ms):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"), "Source": source, "Priority": len(threats),
        "Environment": len(ecology), "Labels": ", ".join(sorted({i["label"] for i in threats + ecology})) or "none",
        "Inference ms": round(ms)})

# ==============================================================================
# TAB 1 - IMAGE
# ==============================================================================
with tab_image:
    img_file = st.file_uploader("Drop reconnaissance imagery", type=["jpg", "jpeg", "png", "webp", "avif"],
                                help="JPG, PNG, WebP or AVIF")
    if img_file is None:
        st.markdown('<div class="card"><h4>No image yet</h4><p class="desc">Upload a frame to run detection, VLM verification, '
                    'and a scene summary. Turn on underwater enhancement in the sidebar for murky footage.</p></div>', unsafe_allow_html=True)
    else:
        data = img_file.getvalue()
        key = f"{hash(data)}_{conf_thresh}_{iou_thresh}_{use_groq_vision}_{GROQ_AVAILABLE}_{enhance}_{max_verify}"
        if st.session_state.get("img_key") != key or "img" not in st.session_state:
            for k in [k for k in st.session_state if k.startswith(("threat_id_", "eco_id_"))]:
                del st.session_state[k]
            st.session_state.pop("summary", None)
            pil = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            work = enhance_underwater(frame) if enhance else frame
            with st.spinner("Running YOLO and VLM verification…"):
                t0 = time.time()
                res = model.predict(source=work, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
                raw_t, raw_e = engine.evaluate_detections(res.boxes, conf_thresh)
                if use_groq_vision and GROQ_AVAILABLE:
                    threats, ecology = split_by_vlm(raw_t + raw_e, work, max_verify)
                    gt, ge = get_groq_supplementary_detections(work)
                    existing = threats + ecology
                    threats += filter_duplicate_boxes(gt, existing)
                    ecology += filter_duplicate_boxes(ge, existing)
                else:
                    threats, ecology = raw_t, raw_e
                threats, ecology = apply_label_filter(threats, label_filter), apply_label_filter(ecology, label_filter)
                ms = (time.time() - t0) * 1000
            if alert_toggle and threats:
                throttled_alert(alerter, threats, frame)
            trigger_threat_alarm(threats, audio_alarm_toggle)
            log_history(img_file.name, threats, ecology, ms)
            st.session_state.update(img_key=key, img=dict(pil=pil, frame=frame, work=work, threats=threats, ecology=ecology, ms=ms))

        S = st.session_state.img
        threats, ecology, frame, work = S["threats"], S["ecology"], S["frame"], S["work"]
        annotated = draw_verified_boxes(work, threats, ecology, thickness, show_conf)
        allc = [i["confidence"] for i in threats + ecology]

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Total contacts", len(allc), "all classes"), unsafe_allow_html=True)
        k2.markdown(kpi("Priority targets", len(threats), "vehicles & divers", danger=bool(threats)), unsafe_allow_html=True)
        k3.markdown(kpi("Avg confidence", f"{np.mean(allc) * 100:.0f}%" if allc else "–", "across contacts"), unsafe_allow_html=True)
        k4.markdown(kpi("Analysis time", f"{S['ms'] / 1000:.1f}s", "YOLO + VLM"), unsafe_allow_html=True)
        st.write("")

        left, right = st.columns([3, 2])
        with left:
            v1, v2, v3 = st.tabs(["Annotated", "Original", "Enhanced"])
            v1.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            v2.image(S["pil"], use_container_width=True)
            v3.image(cv2.cvtColor(enhance_underwater(frame), cv2.COLOR_BGR2RGB), use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Annotated PNG", png_bytes(annotated), "deepaqua_annotated.png", "image/png", use_container_width=True)
            df = to_dataframe(threats, ecology)
            d2.download_button("Detections CSV", df.to_csv(index=False), "detections.csv", "text/csv", use_container_width=True)
            d3.download_button("Detections JSON", df.to_json(orient="records", indent=2), "detections.json", "application/json", use_container_width=True)

        with right:
            st.markdown('<div class="card"><h4>Scene briefing</h4><p class="desc">A short analyst summary from the VLM.</p></div>', unsafe_allow_html=True)
            if st.button("Generate briefing", disabled=not GROQ_AVAILABLE, use_container_width=True):
                with st.spinner("Asking the analyst…"):
                    st.session_state.summary = scene_summary(work)
            if "summary" in st.session_state:
                st.info(st.session_state.summary)

            st.markdown(f"**Priority targets** · {len(threats)}")
            if not threats:
                st.caption("No vehicles or divers in frame.")
            for i, t in enumerate(threats):
                idk = f"threat_id_{i}"
                st.session_state.setdefault(idk, t["label"])
                st.markdown(det_row(t, True, st.session_state[idk]), unsafe_allow_html=True)
                if st.button(f"Identify exact model #{i + 1}", key=f"scan_t_{i}", disabled=not GROQ_AVAILABLE):
                    with st.spinner("Identifying…"):
                        st.session_state[idk] = run_deep_scan_groq(work, t["bbox"], True)
                    st.rerun()

            st.markdown(f"**Environment** · {len(ecology)}")
            if not ecology:
                st.caption("No plants, reefs or structures in frame.")
            for j, e in enumerate(ecology):
                ek = f"eco_id_{j}"
                st.session_state.setdefault(ek, e["label"])
                st.markdown(det_row(e, False, st.session_state[ek]), unsafe_allow_html=True)
                if st.button(f"Identify exact species #{j + 1}", key=f"scan_e_{j}", disabled=not GROQ_AVAILABLE):
                    with st.spinner("Identifying…"):
                        st.session_state[ek] = run_deep_scan_groq(work, e["bbox"], False)
                    st.rerun()

        if not df.empty:
            with st.expander("Detection table and class mix", expanded=False):
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.bar_chart(df["Label"].value_counts())

# ==============================================================================
# TAB 2 - VIDEO
# ==============================================================================
with tab_video:
    c1, c2, c3 = st.columns([2, 1, 1])
    vid_file = c1.file_uploader("Drop patrol footage", type=["mp4", "avi", "mov"])
    water_current_base = c2.number_input("Surface flow baseline (km/h)", 0.0, 45.0, 12.4)
    frame_skip = c3.slider("Analyze every Nth frame", 1, 10, 2, help="Higher is faster on slow machines.")

    if vid_file is None:
        st.markdown('<div class="card"><h4>No footage yet</h4><p class="desc">Upload a clip. Detections run live; the VLM verifies in the background '
                    'every few seconds so playback never stalls.</p></div>', unsafe_allow_html=True)
    else:
        suffix = Path(vid_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(vid_file.getvalue())
        cap = cv2.VideoCapture(tf.name)

        play_feed = st.checkbox("Process feed", value=True)
        col_v1, col_v2 = st.columns([3, 1])
        video_ph = col_v1.empty()
        with col_v2:
            m_fps, m_flow, m_tgt, m_status = st.empty(), st.empty(), st.empty(), st.empty()

        class VLMState:
            is_processing = False
            threats, ecology = [], []
        vlm = VLMState()

        def background_vlm_task(frame_copy, rt, re_):
            try:
                t_out, e_out = split_by_vlm(rt + re_, frame_copy, max_verify)
                gt, ge = get_groq_supplementary_detections(frame_copy)
                ex = t_out + e_out
                t_out += filter_duplicate_boxes(gt, ex); e_out += filter_duplicate_boxes(ge, ex)
                vlm.threats, vlm.ecology = t_out, e_out
            except Exception:
                pass
            finally:
                vlm.is_processing = False

        n, last_t, fps = 0, time.time(), 0.0
        timeline, events, seen, last_annot = [], [], set(), None
        active_t, active_e = [], []
        while cap.isOpened() and play_feed:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            if n % frame_skip:
                continue
            work = enhance_underwater(frame) if enhance else frame
            flow = round(water_current_base + np.sin(n / 10.0) * 1.8, 2)
            res = model.predict(source=work, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
            raw_t, raw_e = engine.evaluate_detections(res.boxes, conf_thresh)

            if use_groq_vision and GROQ_AVAILABLE:
                if n % (15 * frame_skip) == 0 and not vlm.is_processing:
                    vlm.is_processing = True
                    threading.Thread(target=background_vlm_task, args=(work.copy(), raw_t, raw_e), daemon=True).start()
                active_t, active_e = (vlm.threats, vlm.ecology) if (vlm.threats or vlm.ecology) else (raw_t, raw_e)
            else:
                active_t, active_e = raw_t, raw_e
            active_t, active_e = apply_label_filter(active_t, label_filter), apply_label_filter(active_e, label_filter)

            for it in active_t + active_e:
                if it["label"] not in seen:
                    seen.add(it["label"])
                    events.append({"Frame": n, "Label": it["label"], "Group": "Priority" if it in active_t else "Environment",
                                   "Confidence": round(it["confidence"], 3)})
            timeline.append({"frame": n, "Priority": len(active_t), "Environment": len(active_e)})

            ann = draw_verified_boxes(work, active_t, active_e, thickness, show_conf)
            cv2.putText(ann, f"FLOW {flow} KM/H", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if use_groq_vision and vlm.is_processing:
                cv2.putText(ann, "VLM verifying…", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            last_annot = ann
            if alert_toggle and active_t:
                throttled_alert(alerter, active_t, frame)
            trigger_threat_alarm(active_t, audio_alarm_toggle)

            now = time.time(); fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3); last_t = now
            video_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            m_fps.markdown(kpi("Processing", f"{fps:.1f} fps", f"frame {n}"), unsafe_allow_html=True)
            m_flow.markdown(kpi("Flow velocity", f"{flow}", "km/h"), unsafe_allow_html=True)
            m_tgt.markdown(kpi("Targets", len(active_t) + len(active_e), f"{len(active_t)} priority"), unsafe_allow_html=True)
            m_status.markdown(kpi("Status", "RED" if active_t else "CLEAR",
                                  f"{active_t[0]['label']} detected" if active_t else "No hostile contacts", danger=bool(active_t)), unsafe_allow_html=True)
        cap.release()
        try:
            os.unlink(tf.name)
        except OSError:
            pass

        if timeline:
            log_history(vid_file.name, active_t, active_e, 0)
            st.markdown("#### Run summary")
            st.line_chart(pd.DataFrame(timeline).set_index("frame"))
            ev = pd.DataFrame(events)
            if not ev.empty:
                st.dataframe(ev, use_container_width=True, hide_index=True)
                st.download_button("Download event log (CSV)", ev.to_csv(index=False), "video_events.csv", "text/csv")
            if last_annot is not None:
                st.download_button("Download last frame", png_bytes(last_annot), "last_frame.png", "image/png")

# ============================================================================== 
# TAB 3 - BATCH RECON
# ==============================================================================
with tab_batch:
    st.markdown('<div class="card"><h4>Batch reconnaissance</h4><p class="desc">Analyze several still frames in one run and export annotated evidence with a consolidated detection table.</p></div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Add multiple image frames",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        accept_multiple_files=True,
        key="batch_image_upload",
    )
    batch_limit = st.slider("Maximum frames in this run", 1, 20, 8, key="batch_limit")
    batch_vlm = st.toggle("Verify batch detections with the VLM", value=False, disabled=not GROQ_AVAILABLE, key="batch_vlm")

    if st.button("Analyze batch", type="primary", disabled=not batch_files, use_container_width=True):
        batch_rows, archive = [], io.BytesIO()
        batch_priority = []
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for batch_file in batch_files[:batch_limit]:
                try:
                    batch_pil = Image.open(batch_file).convert("RGB")
                    batch_frame = cv2.cvtColor(np.array(batch_pil), cv2.COLOR_RGB2BGR)
                    batch_work = enhance_underwater(batch_frame) if enhance else batch_frame
                    batch_result = model.predict(
                        source=batch_work, conf=conf_thresh, iou=iou_thresh, verbose=False
                    )[0]
                    bt, be = engine.evaluate_detections(batch_result.boxes, conf_thresh)
                    if batch_vlm and GROQ_AVAILABLE:
                        bt, be = split_by_vlm(bt + be, batch_work, max_verify)
                        extra_t, extra_e = get_groq_supplementary_detections(batch_work)
                        existing = bt + be
                        bt += filter_duplicate_boxes(extra_t, existing)
                        be += filter_duplicate_boxes(extra_e, existing)
                    bt = apply_label_filter(bt, label_filter)
                    be = apply_label_filter(be, label_filter)
                    batch_priority.extend(bt)
                    batch_rows.extend(
                        {"File": batch_file.name, **row}
                        for row in to_dataframe(bt, be).to_dict(orient="records")
                    )
                    annotated_batch = draw_verified_boxes(batch_work, bt, be, thickness, show_conf)
                    bundle.writestr(
                        f"{Path(batch_file.name).stem}_annotated.png",
                        png_bytes(annotated_batch),
                    )
                    log_history(batch_file.name, bt, be, 0)
                    if alert_toggle and bt:
                        throttled_alert(alerter, bt, batch_frame)
                except Exception as exc:
                    batch_rows.append({
                        "File": batch_file.name,
                        "Group": "Error",
                        "Label": type(exc).__name__,
                        "Confidence": 0,
                        "Source": "Upload",
                    })
        if batch_priority:
            trigger_threat_alarm(batch_priority, audio_alarm_toggle)
        batch_df = pd.DataFrame(batch_rows)
        st.session_state.batch_results = {
            "rows": batch_rows,
            "csv": batch_df.to_csv(index=False),
            "json": batch_df.to_json(orient="records", indent=2),
            "archive": archive.getvalue(),
        }

    if st.session_state.get("batch_results"):
        result = st.session_state.batch_results
        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True, hide_index=True)
        b1, b2, b3 = st.columns(3)
        b1.download_button("Download batch CSV", result["csv"], "deepaqua_batch.csv", "text/csv", use_container_width=True)
        b2.download_button("Download batch JSON", result["json"], "deepaqua_batch.json", "application/json", use_container_width=True)
        b3.download_button("Annotated evidence ZIP", result["archive"], "deepaqua_evidence.zip", "application/zip", use_container_width=True)

# ==============================================================================
# TAB 4 - SESSION & REPORT
# ==============================================================================

with tab_session:
    st.markdown('<div class="card"><h4>Mission context</h4><p class="desc">Metadata is included in downloadable reports.</p></div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    st.session_state.mission_name = mc1.text_input("Mission name", value=st.session_state.mission_name)
    st.session_state.operator_name = mc2.text_input("Operator", value=st.session_state.operator_name)
    hist = pd.DataFrame(st.session_state.history)
    if hist.empty:
        st.markdown('<div class="card"><h4>Nothing logged yet</h4><p class="desc">Run an image or video analysis and it will appear here.</p></div>', unsafe_allow_html=True)
    else:
        a, b, c = st.columns(3)
        a.markdown(kpi("Scans this session", len(hist)), unsafe_allow_html=True)
        b.markdown(kpi("Priority contacts", int(hist["Priority"].sum()), danger=bool(hist["Priority"].sum())), unsafe_allow_html=True)
        c.markdown(kpi("Environment contacts", int(hist["Environment"].sum())), unsafe_allow_html=True)
        st.write("")
        st.dataframe(hist, use_container_width=True, hide_index=True)

        alert_log_path = Path("runs/alerts/alert_log.jsonl")
        if alert_log_path.exists():
            alert_rows = []
            for line in alert_log_path.read_text(encoding="utf-8").splitlines():
                try:
                    alert_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if alert_rows:
                st.markdown("#### Alert delivery log")
                alert_df = pd.DataFrame(alert_rows)
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download alert log",
                    alert_df.to_csv(index=False),
                    "deepaqua_alerts.csv",
                    "text/csv",
                    use_container_width=True,
                )

        report = [f"# DeepAqua mission report", f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
                  f"- Mission: {st.session_state.mission_name}", f"- Operator: {st.session_state.operator_name or 'Not set'}",
                  f"- Mission profile: {mission_profile}", f"- Model: {weights_name}", f"- Confidence threshold: {conf_thresh:.2f}",
                  f"- VLM verification: {'on' if use_groq_vision and GROQ_AVAILABLE else 'off'}", "",
                  "## Scans", "", hist.to_string(index=False)]
        if "summary" in st.session_state:
            report += ["", "## Latest scene briefing", "", st.session_state.summary]
        r1, r2, _ = st.columns([1, 1, 3])
        r1.download_button("Download report", "\n".join(report), "deepaqua_report.md", "text/markdown", type="primary", use_container_width=True)
        r2.download_button("History CSV", hist.to_csv(index=False), "history.csv", "text/csv", use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

st.markdown('<div style="text-align:center;color:var(--mfg);font-size:.75rem;margin:28px 0 8px">DeepAqua Studio · YOLO + Groq Vision</div>', unsafe_allow_html=True)
