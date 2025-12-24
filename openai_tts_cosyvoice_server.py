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
sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")

# Check if vLLM is requested
USE_VLLM = os.environ.get("COSYVOICE_USE_VLLM", "false").lower() == "true"
USE_TRT = os.environ.get("COSYVOICE_USE_TRT", "false").lower() == "true"
FP16 = os.environ.get("COSYVOICE_FP16", "false").lower() == "true"

# Register vLLM model if needed
if USE_VLLM:
    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        print("✅ vLLM model registered successfully")
    except ImportError as e:
        print(f"❌ Error: vLLM not available. Please install vllm==v0.9.0")
        print(f"   Error details: {e}")
        sys.exit(1)

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError:
    print(
        "Error: Could not import CosyVoice. Make sure you are running this from the directory containing CosyVoice repo."
    )
    sys.exit(1)

# Initialize model
MODEL_DIR = "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
if not os.path.exists(MODEL_DIR):
    print(f"Warning: Model directory {MODEL_DIR} not found.")

print(f"Loading CosyVoice model from {MODEL_DIR}...")
print(f"Configuration: vLLM={USE_VLLM}, TensorRT={USE_TRT}, FP16={FP16}")

# Initialize model with vLLM/TRT support
try:
    cosyvoice_model = AutoModel(
        model_dir=MODEL_DIR, load_vllm=USE_VLLM, load_trt=USE_TRT, fp16=FP16
    )
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Falling back to standard PyTorch backend...")
    cosyvoice_model = AutoModel(model_dir=MODEL_DIR)
    USE_VLLM = False
    USE_TRT = False
    print("✅ Model loaded with standard backend")

# ---------- Text cleaning ----------
_MD_CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_MULTI_WS_RE = re.compile(r"\s+")

_UNIT_MAP = {
    "kb": "kilobytes",
    "mb": "megabytes",
    "gb": "gigabytes",
    "tb": "terabytes",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "ms": "milliseconds",
    "s": "seconds",
    "min": "minutes",
    "h": "hours",
    "cm": "centimeters",
    "mm": "millimeters",
    "m": "meters",
    "km": "kilometers",
    "°c": "degrees celsius",
    "°f": "degrees fahrenheit",
    "%": "percent",
}


def _unit_normalize(text: str) -> str:
    def repl(m):
        num = m.group("num")
        unit = m.group("unit").lower()
        unit = _UNIT_MAP.get(unit, unit)
        return f"{num} {unit}"

    return re.sub(
        r"(?P<num>\d+(\.\d+)?)\s?(?P<unit>kb|mb|gb|tb|hz|khz|mhz|ghz|ms|min|h|s|cm|mm|km|m|°c|°f|%)\b",
        repl,
        text,
        flags=re.IGNORECASE,
    )


def clean_text_for_tts(
    text: str, normalize: bool = True, unit_normalization: bool = True
) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _MD_CODEBLOCK_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    text = _MULTI_WS_RE.sub(" ", text).strip()
    if normalize:
        text = text.replace("…", "...")
        text = text.replace("—", "-").replace("–", "-")
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[‘’]", "'", text)
    if unit_normalization:
        text = _unit_normalize(text)
    return text.strip()


def split_for_tts(text: str, max_chars: int = 350) -> List[str]:
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
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False
    normalization_options: Optional[NormalizationOptions] = None


# ---------- Voice + language instruction system ----------
VOICE_SAMPLES_DIR = "voice_samples"

# Male voice names
MALE_VOICE_NAMES = {
    "michael",
    "facu",
    "faculiado",
    "facundito",
    "facunormal",
    "lucho",
    "onyx",
    "echo",
    "fable",
}

# Voice to language mapping
VOICE_LANGUAGE_MAP = {
    "es": "es",
    "spanish": "es",
    "en": "en",
    "english": "en",
    "alloy": "en",
    "echo": "en",
    "fable": "en",
    "onyx": "en",
    "nova": "en",
    "shimmer": "en",
    "fr": "fr",
    "french": "fr",
    "it": "it",
    "italian": "it",
    "pt": "pt",
    "portuguese": "pt",
    "de": "de",
    "german": "de",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "zh": "zh",
    "chinese": "zh",
}

# Female voice prompts - Short and descriptive (following CosyVoice3 examples)
LANGUAGE_FEMALE_PROMPTS = {
    "es": "Voz femenina argentina, elegante.",
    "en": "Female British voice, elegant.",
    "fr": "Voix féminine française, élégante.",
    "it": "Voce femminile italiana, elegante.",
    "pt": "Voz feminina portuguesa, elegante.",
    "de": "Weibliche deutsche Stimme, elegant.",
    "ja": "上品な日本人の女性の声。",
    "ko": "우아한 한국 여성의 목소리.",
    "zh": "优雅的北京女性声音。",
}

# Male voice prompts
LANGUAGE_MALE_PROMPTS = {
    "es": "Voz masculina argentina, distinguida.",
    "en": "Male British voice, distinguished.",
    "fr": "Voix masculine française, distinguée.",
    "it": "Voce maschile italiana, distinta.",
    "pt": "Voz masculina portuguesa, distinta.",
    "de": "Männliche deutsche Stimme, vornehm.",
    "ja": "日本人の男性の声、独特。",
    "ko": "색다른 한국 남성의 목소리.",
    "zh": "有表现力的北京男性声音。",
}


def discover_voice_samples():
    voice_map = {}
    if os.path.exists(VOICE_SAMPLES_DIR):
        for filename in os.listdir(VOICE_SAMPLES_DIR):
            if filename.endswith((".wav", ".mp3", ".flac")):
                voice_name = os.path.splitext(filename)[0]
                voice_map[voice_name] = os.path.join(VOICE_SAMPLES_DIR, filename)
    return voice_map


def get_voice_file(voice: Optional[str]) -> str:
    available_voices = discover_voice_samples()
    if not voice:
        return available_voices.get("default", "CosyVoice/asset/zero_shot_prompt.wav")
    voice_key = voice.lower()
    if voice_key in available_voices:
        return available_voices[voice_key]
    detected_lang = None
    for lang_code in LANGUAGE_FEMALE_PROMPTS.keys():
        if (
            voice_key.startswith(lang_code)
            or f"-{lang_code}" in voice_key
            or f"_{lang_code}" in voice_key
        ):
            detected_lang = lang_code
            break
    if not detected_lang:
        detected_lang = VOICE_LANGUAGE_MAP.get(voice_key)
    if detected_lang:
        suffixed_voice = f"{voice_key}-{detected_lang}"
        if suffixed_voice in available_voices:
            return available_voices[suffixed_voice]
        if detected_lang in available_voices:
            return available_voices[detected_lang]
    return available_voices.get("default", "CosyVoice/asset/zero_shot_prompt.wav")


def build_prompt_text(voice: Optional[str]) -> str:
    """Build CosyVoice3 instruct prompt with correct system prefix and end token."""
    base_prompt = "You are a helpful assistant."
    if not voice:
        return f"{base_prompt}<|endofprompt|>"

    voice_key = voice.lower()
    is_male = any(name in voice_key for name in MALE_VOICE_NAMES)

    detected_lang = None
    for lang in ["es", "en", "fr", "de", "it", "pt", "ja", "ko", "zh"]:
        if f"-{lang}" in voice_key or voice_key == lang:
            detected_lang = lang
            break
    if not detected_lang:
        detected_lang = VOICE_LANGUAGE_MAP.get(voice_key)

    prompts = LANGUAGE_MALE_PROMPTS if is_male else LANGUAGE_FEMALE_PROMPTS
    instruction = prompts.get(detected_lang, "A natural and professional voice.")

    # FORMAT: "You are a helpful assistant. [Instruction]<|endofprompt|>"
    return f"{base_prompt} {instruction}<|endofprompt|>"


def cosyvoice_generate_wav(
    text: str, voice: Optional[str], speed: float
) -> Tuple[np.ndarray, int]:
    prompt_text = build_prompt_text(voice)
    prompt_wav = get_voice_file(voice)
    audios = []

    # CosyVoice3 uses inference_instruct2 for combining instructions with reference audio
    if hasattr(cosyvoice_model, "inference_instruct2"):
        print(
            f"DEBUG: Using CosyVoice3 inference_instruct2 | Voice: {voice} | Prompt: {prompt_text}"
        )
        for j in cosyvoice_model.inference_instruct2(
            text, prompt_text, prompt_wav, stream=False, speed=speed
        ):
            audios.append(j["tts_speech"].cpu().numpy())
    else:
        # Fallback to zero_shot if instruct2 is not available (though it should be for CosyVoice3)
        print(
            f"DEBUG: Using fallback inference_zero_shot | Voice: {voice} | Prompt: {prompt_text}"
        )
        for j in cosyvoice_model.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=False, speed=speed
        ):
            audios.append(j["tts_speech"].cpu().numpy())

    if not audios:
        raise RuntimeError("No audio generated")
    full_audio = np.concatenate(audios, axis=1).squeeze()
    return full_audio.astype(np.float32), cosyvoice_model.sample_rate


def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    buf = io.BytesIO()
    if fmt in ("wav", "wave"):
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()
    import tempfile, subprocess

    with tempfile.TemporaryDirectory() as td:
        wav_path, out_path = os.path.join(td, "out.wav"), os.path.join(td, f"out.{fmt}")
        sf.write(wav_path, audio, sr, format="WAV")
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
            raise ValueError(f"Unsupported format: {fmt}")
        cmd.append(out_path)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out_path, "rb") as f:
            return f.read()


API_KEY = os.environ.get("TTS_API_KEY", "not-needed")
app = FastAPI(title="CosyVoice3 OpenAI-Compatible TTS")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "cosyvoice3",
        "backend": "vllm" if USE_VLLM else "pytorch",
    }


@app.post("/v1/warmup")
def warmup():
    """Pre-warm the model with a dummy request to ensure it's loaded and cached."""
    try:
        test_text = "Hello world"
        test_voice = "en"
        audio, sr = cosyvoice_generate_wav(test_text, test_voice, speed=1.0)
        return {
            "status": "warmed",
            "model": "cosyvoice3",
            "backend": "vllm" if USE_VLLM else "pytorch",
            "sample_rate": sr,
            "audio_duration_ms": int((len(audio) / sr) * 1000),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "cosyvoice3", "object": "model"}]}


@app.get("/v1/voices")
def list_voices():
    available_voices = discover_voice_samples()
    voices_info = []
    for voice_name, voice_path in available_voices.items():
        language = "en"
        for lang in ["es", "en", "fr", "it", "pt", "de", "ja", "ko", "zh"]:
            if f"-{lang}" in voice_name or voice_name == lang:
                language = lang
                break
        gender = (
            "male"
            if any(name in voice_name.lower() for name in MALE_VOICE_NAMES)
            else "female"
        )
        voices_info.append({"id": voice_name, "language": language, "gender": gender})
    return {"object": "list", "data": voices_info}


@app.post("/v1/audio/speech")
def audio_speech(
    req: SpeechRequest, authorization: Optional[str] = Header(default=None)
):
    if API_KEY and API_KEY not in ["changeme", "not-needed"]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401)
        if authorization.split(" ", 1)[1].strip() != API_KEY:
            raise HTTPException(status_code=401)
    norm = req.normalization_options or NormalizationOptions()
    cleaned = clean_text_for_tts(
        req.input, normalize=norm.normalize, unit_normalization=norm.unit_normalization
    )
    chunks = split_for_tts(cleaned)
    audios, sr = [], None
    for chunk in chunks:
        audio, this_sr = cosyvoice_generate_wav(chunk, req.voice, req.speed)
        if sr is None:
            sr = this_sr
        audios.append(audio)
    if sr is None:
        raise HTTPException(status_code=400, detail="Empty input")
    full = np.concatenate(audios).astype(np.float32)
    payload = encode_audio(full, sr, req.response_format)
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
