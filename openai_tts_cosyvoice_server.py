import io
import os
import re
import sys
import unicodedata
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# Add CosyVoice paths
sys.path.append('CosyVoice')
sys.path.append('CosyVoice/third_party/Matcha-TTS')

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError:
    print("Error: Could not import CosyVoice. Make sure you are running this from the directory containing CosyVoice repo.")
    sys.exit(1)

# Initialize model
# Adjust path to be relative to where we run the server
MODEL_DIR = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B'
if not os.path.exists(MODEL_DIR):
    # Fallback or error
    print(f"Warning: Model directory {MODEL_DIR} not found.")

print(f"Loading CosyVoice model from {MODEL_DIR}...")
# We use the same initialization as in example.py for CosyVoice3
cosyvoice_model = AutoModel(model_dir=MODEL_DIR)
print("Model loaded.")

# ---------- Text cleaning (Kokoro-style intent: sanitize + normalize flags) ----------
_MD_CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_MULTI_WS_RE = re.compile(r"\s+")

_UNIT_MAP = {
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "ms": "milliseconds", "s": "seconds", "min": "minutes", "h": "hours",
    "cm": "centimeters", "mm": "millimeters", "m": "meters", "km": "kilometers",
    "°c": "degrees celsius", "°f": "degrees fahrenheit",
    "%": "percent",
}

def _unit_normalize(text: str) -> str:
    # Examples: 10KB -> "10 kilobytes", 2.4GHz -> "2.4 gigahertz"
    def repl(m):
        num = m.group("num")
        unit = m.group("unit").lower()
        unit = _UNIT_MAP.get(unit, unit)
        return f"{num} {unit}"
    return re.sub(r"(?P<num>\d+(\.\d+)?)\s?(?P<unit>kb|mb|gb|tb|hz|khz|mhz|ghz|ms|min|h|s|cm|mm|km|m|°c|°f|%)\b", repl, text, flags=re.IGNORECASE)

def clean_text_for_tts(
    text: str,
    normalize: bool = True,
    unit_normalization: bool = True,
) -> str:
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalize (helps with weird punctuation/width variants)
    text = unicodedata.normalize("NFKC", text)

    # Remove code blocks (markdown)
    text = _MD_CODEBLOCK_RE.sub(" ", text)

    # Convert markdown links: [label](url) -> label
    text = _MD_LINK_RE.sub(r"\1", text)

    # Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # Remove control chars
    text = _CONTROL_CHARS_RE.sub("", text)

    # Normalize whitespace/newlines
    text = text.replace("\u00A0", " ")
    text = _MULTI_WS_RE.sub(" ", text).strip()

    if normalize:
        # Common punctuation normalizations
        text = text.replace("…", "...")
        text = text.replace("—", "-").replace("–", "-")
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[‘’]", "'", text)

    if unit_normalization:
        text = _unit_normalize(text)

    return text.strip()

def split_for_tts(text: str, max_chars: int = 350) -> List[str]:
    # Simple sentence-ish splitting to avoid very long contexts
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[\.\?\!。！？])\s+", text)
    out, buf = [], ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p.strip()
    if buf:
        out.append(buf)
    return out

# ---------- OpenAI-compatible request schema ----------
class NormalizationOptions(BaseModel):
    normalize: bool = True
    unit_normalization: bool = True

class SpeechRequest(BaseModel):
    model: str = "cosyvoice3"
    input: str
    voice: Optional[str] = None  # you can map this to a speaker/prompt later
    response_format: str = "wav"  # mp3/wav/flac/opus...
    speed: float = 1.0
    stream: bool = False
    normalization_options: Optional[NormalizationOptions] = None

# ---------- CosyVoice inference hook ----------
def cosyvoice_generate_wav(text: str, voice: Optional[str], speed: float) -> Tuple[np.ndarray, int]:
    """
    Return (audio_float32, sample_rate).
    """
    # Default prompt
    prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
    prompt_wav = 'CosyVoice/asset/zero_shot_prompt.wav'
    
    # If voice is provided and looks like a file, use it?
    # Or maybe map "alloy" to something?
    # For now, we stick to the default zero-shot prompt from example.py
    
    # Run inference
    # inference_zero_shot yields results. We collect them.
    # Note: speed parameter is supported in CosyVoice3 inference_zero_shot
    
    audios = []
    for j in cosyvoice_model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False, speed=speed):
        audios.append(j['tts_speech'].cpu().numpy())
    
    if not audios:
        raise RuntimeError("No audio generated by CosyVoice")
        
    # Concatenate if multiple chunks (though usually one for short text)
    full_audio = np.concatenate(audios, axis=1) # shape (1, N)
    
    # Squeeze to 1D array
    full_audio = full_audio.squeeze()
    
    return full_audio.astype(np.float32), cosyvoice_model.sample_rate

def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    buf = io.BytesIO()
    if fmt in ("wav", "wave"):
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()

    # For mp3/opus/aac: easiest is ffmpeg (pydub or direct subprocess).
    # Keep it simple: write wav then transcode with ffmpeg.
    import tempfile, subprocess

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "out.wav")
        out_path = os.path.join(td, f"out.{fmt}")
        sf.write(wav_path, audio, sr, format="WAV")

        # Basic ffmpeg transcode
        cmd = ["ffmpeg", "-y", "-i", wav_path]
        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
        elif fmt == "opus":
            cmd += ["-codec:a", "libopus", "-b:a", "96k"]
        elif fmt == "aac":
            cmd += ["-codec:a", "aac", "-b:a", "128k"]
        elif fmt == "flac":
            cmd += ["-codec:a", "flac"]
        else:
            raise ValueError(f"Unsupported response_format: {fmt}")

        cmd.append(out_path)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(out_path, "rb") as f:
            return f.read()

# ---------- FastAPI app ----------
API_KEY = os.environ.get("TTS_API_KEY", "not-needed")

app = FastAPI(title="CosyVoice3 OpenAI-Compatible TTS")

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "cosyvoice3", "object": "model"}]}

@app.post("/v1/audio/speech")
def audio_speech(req: SpeechRequest, authorization: Optional[str] = Header(default=None)):
    # Simple API key check (Open-WebUI will send: Authorization: Bearer <key>)
    # If API_KEY is "not-needed", we skip the check.
    if API_KEY and API_KEY not in ["changeme", "not-needed"]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    norm = req.normalization_options or NormalizationOptions()
    cleaned = clean_text_for_tts(req.input, normalize=norm.normalize, unit_normalization=norm.unit_normalization)
    chunks = split_for_tts(cleaned)

    # Generate audio per chunk and concatenate
    audios = []
    sr = None
    for chunk in chunks:
        audio, this_sr = cosyvoice_generate_wav(chunk, req.voice, req.speed)
        if sr is None:
            sr = this_sr
        elif sr != this_sr:
            raise HTTPException(status_code=500, detail="Sample rate mismatch between chunks")
        audios.append(audio)

    if sr is None:
        raise HTTPException(status_code=400, detail="Empty input after cleaning")

    full = np.concatenate(audios).astype(np.float32)

    try:
        payload = encode_audio(full, sr, req.response_format)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    media = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }.get(req.response_format.lower(), "application/octet-stream")

    if req.stream:
        return StreamingResponse(io.BytesIO(payload), media_type=media)
    return Response(content=payload, media_type=media)
