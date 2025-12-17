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

try:
    from cosyvoice.cli.cosyvoice import AutoModel
except ImportError:
    print(
        "Error: Could not import CosyVoice. Make sure you are running this from the directory containing CosyVoice repo."
    )
    sys.exit(1)

# Initialize model
# Adjust path to be relative to where we run the server
MODEL_DIR = "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
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
    # Examples: 10KB -> "10 kilobytes", 2.4GHz -> "2.4 gigahertz"
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
    text = text.replace("\u00a0", " ")
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
# ---------- Voice + language instruction system ----------
# Voice samples directory
VOICE_SAMPLES_DIR = "voice_samples"

# Female voice prompts - Capital cities used as accent/style anchors
LANGUAGE_FEMALE_PROMPTS = {
    "es": "Eres una mujer de Buenos Aires, Argentina. Tu voz es femenina, elegante, carismática y muy expresiva. Habla con pasión y calidez, como una actriz profesional argentina.",
    "en": "You are a woman from London, England. Your voice is female, posh, charismatic, and highly expressive. Speak with refined British elegance and captivating warmth, like a professional British actress.",
    "fr": "Tu es une femme de Paris, France. Ta voix est féminine, élégante, charismatique et très expressive. Parle avec sophistication et charme parisien, comme une actrice professionnelle française.",
    "it": "Sei una donna di Roma, Italia. La tua voce è femminile, elegante, carismatica e molto espressiva. Parla con passione e calore romano, come un'attrice professionista italiana.",
    "pt": "És uma mulher de Lisboa, Portugal. A tua voz é feminina, elegante, carismática e muito expressiva. Fala com sofisticação e calor lisboeta, como uma atriz profissional portuguesa.",
    "de": "Du bist eine Frau aus Berlin, Deutschland. Deine Stimme ist weiblich, elegant, charismatisch und sehr ausdrucksstark. Sprich mit Berliner Raffinesse und Wärme, wie eine professionelle deutsche Schauspielerin.",
    "ja": "あなたは東京出身の女性です。あなたの声は女性的で、上品で、カリスマ的で、非常に表現豊かです。プロの日本人女優のように、洗練された東京のエレガンスと魅力的な温かさで話してください。",
    "ko": "당신은 서울 출신 여성입니다. 당신의 목소리는 여성스럽고, 우아하며, 카리스마 있고, 매우 표현력이 풍부합니다. 전문 한국 배우처럼 세련된 서울의 우아함과 매력적인 따뜻함으로 말하세요.",
    "zh": "你是来自北京的女性。你的声音是女性化的、优雅的、有魅力的、非常有表现力的。请像专业的中国女演员一样，以精致的北京优雅和迷人的温暖说话。",
}

# Male voice prompts - Capital cities used as accent/style anchors
LANGUAGE_MALE_PROMPTS = {
    "es": "Eres un hombre de Buenos Aires, Argentina. Tu voz es masculina, elegante, carismática y muy expresiva. Habla con pasión y calidez, como un actor profesional argentino con clase y distinción.",
    "en": "You are a man from London, England. Your voice is male, posh, charismatic, and highly expressive. Speak with refined British elegance and captivating warmth, like a distinguished British gentleman and professional actor.",
    "fr": "Tu es un homme de Paris, France. Ta voix est masculine, élégante, charismatique et très expressive. Parle avec sophistication et charme parisien, comme un acteur professionnel français.",
    "it": "Sei un uomo di Roma, Italia. La tua voce è maschile, elegante, carismatica e molto espressiva. Parla con passione e calore romano, come un attore professionista italiano.",
    "pt": "És um homem de Lisboa, Portugal. A tua voz é masculina, elegante, carismática e muito expressiva. Fala com sofisticação e calor lisboeta, como um ator profissional português.",
    "de": "Du bist ein Mann aus Berlin, Deutschland. Deine Stimme ist männlich, elegant, charismatisch und sehr ausdrucksstark. Sprich mit Berliner Raffinesse und Wärme, wie ein professioneller deutscher Schauspieler.",
    "ja": "あなたは東京出身の男性です。あなたの声は男性的で、上品で、カリスマ的で、非常に表現豊かです。プロの日本人俳優のように、洗練された東京のエレガンスと魅力的な温かさで話してください。",
    "ko": "당신은 서울 출신 남성입니다. 당신의 목소리는 남성스럽고, 우아하며, 카리스마 있고, 매우 표현력이 풍부합니다. 전문 한국 배우처럼 세련된 서울의 우아함과 매력적인 따뜻함으로 말하세요.",
    "zh": "你是来自北京的男性。你的声音是男性化的、优雅的、有魅力的、非常有表现力的。请像专业的中国男演员一样，以精致的北京优雅和迷人的温暖说话。",
}

# For backward compatibility
LANGUAGE_CAPITAL_INSTRUCTIONS = LANGUAGE_FEMALE_PROMPTS

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
    "multilingual": "en",
    "default": "en",
}

VOICE_BASE_INSTRUCTION = "You are a professional voice actor."


def discover_voice_samples():
    """Discover available voice samples from the voice_samples directory."""
    voice_map = {}
    if os.path.exists(VOICE_SAMPLES_DIR):
        for filename in os.listdir(VOICE_SAMPLES_DIR):
            if filename.endswith((".wav", ".mp3", ".flac")):
                voice_name = os.path.splitext(filename)[0]
                voice_map[voice_name] = os.path.join(VOICE_SAMPLES_DIR, filename)
    return voice_map


def get_voice_file(voice: Optional[str]) -> str:
    """Get the voice file path for a given voice name."""
    # Discover available voices
    available_voices = discover_voice_samples()

    if not voice:
        # Default voice
        if "default" in available_voices:
            return available_voices["default"]
        return "CosyVoice/asset/zero_shot_prompt.wav"

    voice_key = voice.lower()

    # Direct match (exact name)
    if voice_key in available_voices:
        return available_voices[voice_key]

    # Detect target language from voice parameter or mapping
    detected_lang = None

    # Check if voice already has language suffix
    if voice_key.endswith("-es") or voice_key.endswith("-en"):
        if voice_key in available_voices:
            return available_voices[voice_key]
        # Extract base name
        base_voice = voice_key.rsplit("-", 1)[0]
        if base_voice in available_voices:
            return available_voices[base_voice]

    # Try to match by language code in voice name
    for lang_code in LANGUAGE_CAPITAL_INSTRUCTIONS.keys():
        if (
            voice_key.startswith(lang_code)
            or f"-{lang_code}" in voice_key
            or f"_{lang_code}" in voice_key
        ):
            detected_lang = lang_code
            break

    # Check VOICE_LANGUAGE_MAP
    if voice_key in VOICE_LANGUAGE_MAP:
        detected_lang = VOICE_LANGUAGE_MAP[voice_key]

    # Try to find voice with language suffix
    if detected_lang:
        # Try voice_name-lang (e.g., "aimee-en")
        suffixed_voice = f"{voice_key}-{detected_lang}"
        if suffixed_voice in available_voices:
            return available_voices[suffixed_voice]

    # Try language-specific voice files
    if detected_lang and detected_lang in available_voices:
        return available_voices[detected_lang]

    # Fallback to multilingual or default
    if "multilingual" in available_voices:
        return available_voices["multilingual"]
    if "default" in available_voices:
        return available_voices["default"]

    return "CosyVoice/asset/zero_shot_prompt.wav"


def build_prompt_text(voice: Optional[str]) -> str:
    """Build prompt text for CosyVoice3 following best practices.

    According to CosyVoice3 documentation, prompt_text should be a SHORT descriptive
    phrase about the voice style (e.g., "A warm, professional tone").
    The <|endofprompt|> token separates this description from the actual TTS text.
    """
    if not voice:
        return "A natural and professional voice.<|endofprompt|>"

    voice_key = voice.lower()

    # Detect gender
    is_male = False
    voice_base = (
        voice_key.replace("-es", "")
        .replace("-en", "")
        .replace("-fr", "")
        .replace("-de", "")
    )
    for male_name in MALE_VOICE_NAMES:
        if male_name in voice_base:
            is_male = True
            break

    # Detect language
    detected_lang = None
    if "-es" in voice_key or voice_key == "es":
        detected_lang = "es"
    elif "-en" in voice_key or voice_key == "en":
        detected_lang = "en"
    elif "-fr" in voice_key or voice_key == "fr":
        detected_lang = "fr"
    elif "-de" in voice_key or voice_key == "de":
        detected_lang = "de"
    elif "-it" in voice_key or voice_key == "it":
        detected_lang = "it"
    elif "-pt" in voice_key or voice_key == "pt":
        detected_lang = "pt"
    elif "-ja" in voice_key or voice_key == "ja":
        detected_lang = "ja"
    elif "-ko" in voice_key or voice_key == "ko":
        detected_lang = "ko"
    elif "-zh" in voice_key or voice_key == "zh":
        detected_lang = "zh"
    elif voice_key in VOICE_LANGUAGE_MAP:
        detected_lang = VOICE_LANGUAGE_MAP[voice_key]

    # Build appropriate short descriptive phrase
    if detected_lang == "es":
        if is_male:
            return "Una voz masculina argentina, cálida y profesional.<|endofprompt|>"
        else:
            return "Una voz femenina argentina, elegante y profesional.<|endofprompt|>"
    elif detected_lang == "en":
        if is_male:
            return "A male British voice, warm and professional.<|endofprompt|>"
        else:
            return "A female British voice, elegant and professional.<|endofprompt|>"
    elif detected_lang == "fr":
        if is_male:
            return "Une voix masculine française, chaleureuse et professionnelle.<|endofprompt|>"
        else:
            return "Une voix féminine française, élégante et professionnelle.<|endofprompt|>"
    elif detected_lang == "de":
        if is_male:
            return (
                "Eine männliche deutsche Stimme, warm und professionell.<|endofprompt|>"
            )
        else:
            return "Eine weibliche deutsche Stimme, elegant und professionell.<|endofprompt|>"
    else:
        # Default for other languages
        return "A natural and professional voice.<|endofprompt|>"


def cosyvoice_generate_wav(
    text: str, voice: Optional[str], speed: float
) -> Tuple[np.ndarray, int]:
    """
    Return (audio_float32, sample_rate).
    """
    # Build dynamic instruction prompt based on voice/language
    prompt_text = build_prompt_text(voice)

    # Get the appropriate voice file based on voice name/language
    prompt_wav = get_voice_file(voice)

    # Run inference
    # inference_zero_shot yields results. We collect them.
    # Note: speed parameter is supported in CosyVoice3 inference_zero_shot

    audios = []
    for j in cosyvoice_model.inference_zero_shot(
        text, prompt_text, prompt_wav, stream=False, speed=speed
    ):
        audios.append(j["tts_speech"].cpu().numpy())

    if not audios:
        raise RuntimeError("No audio generated by CosyVoice")

    # Concatenate if multiple chunks (though usually one for short text)
    full_audio = np.concatenate(audios, axis=1)  # shape (1, N)

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
    import tempfile
    import subprocess

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


@app.get("/health")
def health():
    return {"status": "ok", "model": "cosyvoice3"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "cosyvoice3", "object": "model"}]}


@app.get("/v1/voices")
def list_voices():
    """List all available voice samples with metadata."""
    available_voices = discover_voice_samples()

    voices_info = []
    for voice_name, voice_path in available_voices.items():
        # Extract language from filename
        language = "unknown"
        if "-es" in voice_name:
            language = "es"
        elif "-en" in voice_name:
            language = "en"
        elif "-fr" in voice_name:
            language = "fr"
        elif "-it" in voice_name:
            language = "it"
        elif "-pt" in voice_name:
            language = "pt"
        elif "-de" in voice_name:
            language = "de"
        elif "-ja" in voice_name:
            language = "ja"
        elif "-ko" in voice_name:
            language = "ko"
        elif "-zh" in voice_name:
            language = "zh"
        elif voice_name in ["es", "en", "fr", "it", "pt", "de", "ja", "ko", "zh"]:
            language = voice_name

        # Detect gender
        gender = "female"
        voice_base = voice_name.replace("-es", "").replace("-en", "")
        for male_name in MALE_VOICE_NAMES:
            if male_name in voice_base:
                gender = "male"
                break

        # Get file size
        file_size = 0
        if os.path.exists(voice_path):
            file_size = os.path.getsize(voice_path)

        voices_info.append(
            {
                "id": voice_name,
                "language": language,
                "gender": gender,
                "file_path": voice_path,
                "file_size": file_size,
            }
        )

    # Sort by language then name
    voices_info.sort(key=lambda x: (x["language"], x["id"]))

    return {"object": "list", "data": voices_info, "total": len(voices_info)}


@app.post("/v1/audio/speech")
def audio_speech(
    req: SpeechRequest, authorization: Optional[str] = Header(default=None)
):
    # Simple API key check (Open-WebUI will send: Authorization: Bearer <key>)
    # If API_KEY is "not-needed", we skip the check.
    if API_KEY and API_KEY not in ["changeme", "not-needed"]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    norm = req.normalization_options or NormalizationOptions()
    cleaned = clean_text_for_tts(
        req.input, normalize=norm.normalize, unit_normalization=norm.unit_normalization
    )
    chunks = split_for_tts(cleaned)

    # Generate audio per chunk and concatenate
    audios = []
    sr = None
    for chunk in chunks:
        audio, this_sr = cosyvoice_generate_wav(chunk, req.voice, req.speed)
        if sr is None:
            sr = this_sr
        elif sr != this_sr:
            raise HTTPException(
                status_code=500, detail="Sample rate mismatch between chunks"
            )
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
