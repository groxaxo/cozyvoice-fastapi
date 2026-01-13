import io
import logging
import os
import re
import sys
import unicodedata
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add CosyVoice paths
sys.path.append("CosyVoice")
sys.path.append("CosyVoice/third_party/Matcha-TTS")

# Check if vLLM/TensorRT is requested
USE_VLLM = os.environ.get("COSYVOICE_USE_VLLM", "false").lower() == "true"
USE_TRT = os.environ.get("COSYVOICE_USE_TRT", "false").lower() == "true"
FP16 = os.environ.get("COSYVOICE_FP16", "false").lower() == "true"

# Use ONNX for the Flow and Hift modules by default.
# When enabled, the server will attempt to load ONNX versions of the
# CosyVoice flow and hift modules to speed up inference. Set
# COSYVOICE_USE_ONNX=false to disable this and fall back to the original
# PyTorch implementation.
USE_ONNX = os.environ.get("COSYVOICE_USE_ONNX", "true").lower() == "true"

# The Hugging Face repository containing the ONNX components.
# Override via COSYVOICE_ONNX_REPO if you wish to use a different repo.
ONNX_REPO = os.environ.get(
    "COSYVOICE_ONNX_REPO", "Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX"
)

# Quantization configuration
QUANTIZATION_ENABLED = os.environ.get("QUANTIZATION_ENABLED", "false").lower() == "true"
QUANTIZATION_BITS = int(os.environ.get("QUANTIZATION_BITS", "4"))  # 4 or 8

# Register vLLM model if needed
if USE_VLLM:
    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        logger.info("vLLM model registered successfully")
    except ImportError as e:
        logger.error("vLLM not available. Please install vllm==v0.9.0")
        logger.debug(f"Import error details: {e}")
        sys.exit(1)

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError:
    logger.error("Could not import CosyVoice. Make sure you are running this from the directory containing CosyVoice repo.")
    sys.exit(1)

# Import quantization libraries if needed
bnb_config = None
quantization_active = False  # Track actual quantization status separately from environment setting

if QUANTIZATION_ENABLED:
    try:
        import torch
        from transformers import BitsAndBytesConfig

        if QUANTIZATION_BITS == 4:
            logger.info("Configuring 4-bit quantization (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            quantization_active = True
        elif QUANTIZATION_BITS == 8:
            logger.info("Configuring 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            quantization_active = True
        else:
            logger.warning(f"Invalid QUANTIZATION_BITS value: {QUANTIZATION_BITS}. Must be 4 or 8. Disabling quantization.")
    except ImportError as e:
        logger.warning("BitsAndBytes or transformers not available. Install with: pip install bitsandbytes>=0.41.0 transformers>=4.48.0")
        logger.debug(f"Import error: {e}")
        bnb_config = None

# Initialize model directory
MODEL_DIR = "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
if not os.path.exists(MODEL_DIR):
    logger.warning(f"Model directory not found at expected path")


def ensure_onnx_models(model_dir: str) -> None:
    """Ensure ONNX flow and hift modules are available in the model directory.

    If USE_ONNX is False, this is a no-op. When enabled, this function checks
    whether the required ONNX files (flow_fp16/flow_fp32 and hift.onnx) are
    present in the given model_dir. If they are missing, it attempts to download
    them from the Hugging Face repository specified by ONNX_REPO using the
    huggingface_hub library. If the download fails or the library is not
    available, the function logs a warning and leaves it up to the caller to
    place the files manually.
    """
    # Do nothing if ONNX is disabled
    if not USE_ONNX:
        return
    file_flow = "flow_fp16.onnx" if FP16 else "flow_fp32.onnx"
    file_hift = "hift.onnx"
    path_flow = os.path.join(model_dir, file_flow)
    path_hift = os.path.join(model_dir, file_hift)
    if os.path.exists(path_flow) and os.path.exists(path_hift):
        logger.info("ONNX flow/hift models found in model directory.")
        return
    try:
        # Import huggingface_hub lazily to avoid requiring it when ONNX is disabled
        from huggingface_hub import hf_hub_download  # type: ignore

        logger.info(
            f"Downloading ONNX models ({file_flow}, {file_hift}) from Hugging Face repo {ONNX_REPO}..."
        )
        hf_hub_download(
            repo_id=ONNX_REPO,
            filename=file_flow,
            local_dir=model_dir,
            cache_dir=model_dir,
        )
        hf_hub_download(
            repo_id=ONNX_REPO,
            filename=file_hift,
            local_dir=model_dir,
            cache_dir=model_dir,
        )
        logger.info("ONNX models downloaded successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to download ONNX models: {e}. Please download them manually and place them in {model_dir}."
        )


logger.info("Loading CosyVoice model...")
logger.info(
    f"Configuration: vLLM={USE_VLLM}, TensorRT={USE_TRT}, FP16={FP16}, Quantization={quantization_active}, ONNX={USE_ONNX}"
)
if quantization_active:
    logger.info(f"Quantization: {QUANTIZATION_BITS}-bit enabled")

# Ensure ONNX models are present before loading the model
ensure_onnx_models(MODEL_DIR)

# Initialize model with vLLM/TRT/Quantization/ONNX support
try:
    # Note: CosyVoice AutoModel may not natively support quantization_config
    # We'll attempt to pass it, but may need to apply quantization differently
    model_kwargs = {
        "model_dir": MODEL_DIR,
        "load_vllm": USE_VLLM,
        "load_trt": USE_TRT,
        "fp16": FP16,
        # Always include load_onnx so that the caller can enable/disable via env var
        "load_onnx": USE_ONNX,
    }

    # Add quantization config if enabled and not using vLLM/TRT
    # (vLLM has its own quantization methods)
    if quantization_active and bnb_config and not USE_VLLM and not USE_TRT:
        logger.info("Attempting to load model with BitsAndBytes quantization...")
        # Try to pass quantization_config to AutoModel
        # This may require modification of CosyVoice's AutoModel class
        try:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            # torch may not be imported yet; import lazily
            import torch  # type: ignore
            model_kwargs["torch_dtype"] = torch.bfloat16
            cosyvoice_model = AutoModel(**model_kwargs)
            logger.info(f"Model loaded successfully with {QUANTIZATION_BITS}-bit quantization")
        except TypeError:
            # AutoModel doesn't support quantization_config parameter
            logger.warning("CosyVoice AutoModel doesn't support quantization_config parameter directly")
            logger.info("Loading model normally - post-load quantization not yet implemented")
            model_kwargs.pop("quantization_config", None)
            model_kwargs.pop("device_map", None)
            model_kwargs.pop("torch_dtype", None)
            cosyvoice_model = AutoModel(**model_kwargs)
            quantization_active = False  # Update module-level flag: quantization not actually active
            logger.info("Model loaded without quantization")
    else:
        cosyvoice_model = AutoModel(**model_kwargs)
        logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model with specified backend: {e}")
    logger.info("Falling back to standard PyTorch backend...")
    try:
        cosyvoice_model = AutoModel(model_dir=MODEL_DIR)
        USE_VLLM = False
        USE_TRT = False
        quantization_active = False  # Update module-level flag: quantization not active in fallback
        logger.info("Model loaded with standard backend")
    except Exception as fallback_err:
        logger.error(f"Failed to load model even with fallback: {fallback_err}")
        raise

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


def discover_voice_samples() -> dict:
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
    audios: List[np.ndarray] = []

    # CosyVoice3 uses inference_instruct2 for combining instructions with reference audio
    if hasattr(cosyvoice_model, "inference_instruct2"):
        logger.debug("Using CosyVoice3 inference_instruct2")
        for j in cosyvoice_model.inference_instruct2(
            text, prompt_text, prompt_wav, stream=False, speed=speed
        ):
            audios.append(j["tts_speech"].cpu().numpy())
    else:
        # Fallback to zero_shot if instruct2 is not available (though it should be for CosyVoice3)
        logger.debug("Using fallback inference_zero_shot")
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
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "out.wav")
        out_path = os.path.join(td, f"out.{fmt}")
        sf.write(wav_path, audio, sr, format="WAV")
        cmd: List[str] = ["ffmpeg", "-y", "-i", wav_path]
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


@app.get("/", response_class=HTMLResponse)
def index():
    """Attractive landing page for the CosyVoice FastAPI server.

    Note: HTML template is inline for single-file deployment simplicity
    and to avoid external dependencies. This makes the server easier to
    distribute and deploy as a standalone file. Dynamic values are
    sanitized as a defensive measure against potential injection attacks.
    """
    available_voices = discover_voice_samples()
    voice_count = len(available_voices)
    # Sanitize values to prevent any potential HTML injection (defensive programming)
    backend = "vLLM" if USE_VLLM else "PyTorch"
    safe_backend = str(backend).replace('<', '&lt;').replace('>', '&gt;')
    safe_voice_count = int(voice_count)  # Ensure it's an integer

    # Add quantization info to badges
    quant_badge = ""
    if quantization_active:
        quant_badge = f'<span class="badge">Quantization: {QUANTIZATION_BITS}-bit</span>'

    # Add ONNX info to badges
    onnx_badge = ""
    if USE_ONNX:
        onnx_badge = '<span class="badge">ONNX enabled</span>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CosyVoice3 TTS Server</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                color: white;
                margin-bottom: 40px;
                padding: 40px 20px;
            }}
            .header h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            .header p {{
                font-size: 1.3em;
                opacity: 0.95;
            }}
            .badge {{
                display: inline-block;
                background: rgba(255,255,255,0.2);
                padding: 8px 16px;
                border-radius: 20px;
                margin: 10px 5px;
                font-size: 0.9em;
                backdrop-filter: blur(10px);
            }}
            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .card h2 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            .card p {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 15px;
            }}
            .code-block {{
                background: #f5f5f5;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                overflow-x: auto;
                margin: 15px 0;
            }}
            .feature-list {{
                list-style: none;
                padding: 0;
            }}
            .feature-list li {{
                padding: 8px 0;
                color: #555;
            }}
            .feature-list li:before {{
                content: "✓ ";
                color: #667eea;
                font-weight: bold;
                margin-right: 8px;
            }}
            .endpoint {{
                background: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-family: monospace;
            }}
            .method {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                margin-right: 10px;
                color: white;
            }}
            .get {{ background: #61affe; }}
            .post {{ background: #49cc90; }}
            .footer {{
                text-align: center;
                color: white;
                padding: 20px;
                margin-top: 40px;
            }}
            a {{
                color: #667eea;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ CosyVoice3 TTS Server</h1>
                <p>OpenAI-Compatible Text-to-Speech API</p>
                <div>
                    <span class="badge">Backend: {safe_backend}</span>
                    <span class="badge">Voices: {safe_voice_count}</span>
                    <span class="badge">Model: Fun-CosyVoice3-0.5B-2512</span>
                    {quant_badge}
                    {onnx_badge}
                </div>
            </div>

            <div class="cards">
                <div class="card">
                    <h2>🚀 Quick Start</h2>
                    <p>Generate speech using the OpenAI-compatible API:</p>
                    <div class="code-block">
curl -X POST http://localhost:8000/v1/audio/speech \\\n+  -H "Content-Type: application/json" \\\n+  -d '{{'<br>
    "model": "cosyvoice3",<br>
    "input": "Hello, world!",<br>
    "voice": "en"<br>
  }}' -o output.wav
                    </div>
                </div>

                <div class="card">
                    <h2>✨ Features</h2>
                    <ul class="feature-list">
                        <li>OpenAI-compatible API endpoints</li>
                        <li>Multi-language support (9+ languages)</li>
                        <li>Zero-shot voice cloning</li>
                        <li>Multiple audio formats (WAV, MP3, FLAC, etc.)</li>
                        <li>Speed control</li>
                        <li>Streaming support</li>
                        <li>Custom voice samples</li>
                    </ul>
                </div>

                <div class="card">
                    <h2>📡 API Endpoints</h2>
                    <div class="endpoint">
                        <span class="method get">GET</span> /health
                    </div>
                    <div class="endpoint">
                        <span class="method get">GET</span> /v1/models
                    </div>
                    <div class="endpoint">
                        <span class="method get">GET</span> /v1/voices
                    </div>
                    <div class="endpoint">
                        <span class="method post">POST</span> /v1/audio/speech
                    </div>
                    <div class="endpoint">
                        <span class="method post">POST</span> /v1/warmup
                    </div>
                    <p style="margin-top: 15px;">
                        <a href="/docs" target="_blank">📖 View Full API Documentation</a>
                    </p>
                </div>
            </div>

            <div class="cards">
                <div class="card">
                    <h2>🎭 Available Voices</h2>
                    <p>The server currently has <strong>{safe_voice_count}</strong> voice sample(s) available.</p>
                    <p style="margin-top: 10px;">
                        <a href="/v1/voices">View all available voices →</a>
                    </p>
                    <p style="margin-top: 15px; color: #888; font-size: 0.9em;">
                        Add custom voices to the <code>voice_samples/</code> directory to expand voice options.
                    </p>
                </div>

                <div class="card">
                    <h2>🔧 Integration</h2>
                    <p>Use with Open-WebUI or any OpenAI-compatible client:</p>
                    <ul class="feature-list" style="margin-top: 15px;">
                        <li><strong>API Base URL:</strong> http://your-server:8000/v1</li>
                        <li><strong>Model:</strong> cosyvoice3</li>
                        <li><strong>API Key:</strong> not-needed (or configured)</li>
                    </ul>
                </div>

                <div class="card">
                    <h2>📚 Documentation</h2>
                    <p>Comprehensive guides and references:</p>
                    <ul class="feature-list" style="margin-top: 15px;">
                        <li><a href="/docs">Interactive API Documentation</a></li>
                        <li><a href="/redoc">ReDoc API Reference</a></li>
                        <li>README.md - Setup & Configuration</li>
                        <li>Voice Mapping Guide</li>
                    </ul>
                </div>
            </div>

            <div class="footer">
                <p>Powered by CosyVoice3 • FastAPI • {safe_backend} Backend</p>
                <p style="margin-top: 10px; opacity: 0.8;">
                    High-quality multilingual text-to-speech with zero-shot voice cloning
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
def health():
    health_info = {
        "status": "ok",
        "model": "cosyvoice3",
        "backend": "vllm" if USE_VLLM else "pytorch",
    }
    if quantization_active:
        health_info["quantization"] = f"{QUANTIZATION_BITS}-bit"
    health_info["onnx"] = USE_ONNX
    return health_info


@app.post("/v1/warmup")
def warmup():
    """Pre-warm the model with a dummy request to ensure it's loaded and cached."""
    try:
        test_text = "Hello world"
        test_voice = "en"
        audio, sr = cosyvoice_generate_wav(test_text, test_voice, speed=1.0)
        warmup_info = {
            "status": "warmed",
            "model": "cosyvoice3",
            "backend": "vllm" if USE_VLLM else "pytorch",
            "sample_rate": sr,
            "audio_duration_ms": int((len(audio) / sr) * 1000),
        }
        if quantization_active:
            warmup_info["quantization"] = f"{QUANTIZATION_BITS}-bit"
        warmup_info["onnx"] = USE_ONNX
        return warmup_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "cosyvoice3", "object": "model"}]}


@app.get("/v1/voices")
def list_voices():
    available_voices = discover_voice_samples()
    voices_info: List[dict] = []
    for voice_name, _voice_path in available_voices.items():
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
    audios: List[np.ndarray] = []
    sr: Optional[int] = None
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