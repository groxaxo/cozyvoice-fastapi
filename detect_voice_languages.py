#!/usr/bin/env python3
"""
Detect language of voice samples using Whisper and rename with language suffix.
"""

import os
import json
import requests
import subprocess
from pathlib import Path

WHISPER_API_URL = "http://100.85.200.52:8887/v1/audio/transcriptions"
VOICE_SAMPLES_DIR = "/home/op/cozyvoice_fastapi/voice_samples"

# Known voice mappings based on names
KNOWN_VOICES = {
    "lucho": "es",  # Latino from Argentina
    "facu": "es",  # Latino from Argentina
    "faculiado": "es",
    "facundito": "es",
    "facunormal": "es",
    "vozespanola": "es",
    "story_spanish": "es",
    "colombiana": "es",
    "se_fue_alabosta": "es",
}


def detect_language_whisper(audio_file_path):
    """Detect language using Whisper API."""
    try:
        with open(audio_file_path, "rb") as f:
            files = {"file": (os.path.basename(audio_file_path), f, "audio/mpeg")}
            data = {"model": "whisper-1"}

            response = requests.post(
                WHISPER_API_URL, files=files, data=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            text = result.get("text", "")

            # Simple language detection based on text
            if not text:
                return None

            # Check for Spanish indicators
            spanish_words = [
                "el",
                "la",
                "de",
                "que",
                "y",
                "en",
                "un",
                "una",
                "es",
                "por",
                "para",
                "con",
                "no",
                "se",
                "los",
                "las",
            ]
            english_words = [
                "the",
                "and",
                "is",
                "to",
                "in",
                "of",
                "that",
                "it",
                "for",
                "on",
                "with",
                "as",
                "was",
                "at",
            ]

            text_lower = text.lower()
            spanish_count = sum(
                1 for word in spanish_words if f" {word} " in f" {text_lower} "
            )
            english_count = sum(
                1 for word in english_words if f" {word} " in f" {text_lower} "
            )

            print(f"  Transcription: {text[:100]}...")
            print(
                f"  Spanish markers: {spanish_count}, English markers: {english_count}"
            )

            if spanish_count > english_count:
                return "es"
            elif english_count > spanish_count:
                return "en"
            else:
                return None

    except Exception as e:
        print(f"  Error detecting language: {e}")
        return None


def process_voice_files():
    """Process all MP3 files in voice_samples directory."""
    results = {}

    # Get all MP3 files
    mp3_files = list(Path(VOICE_SAMPLES_DIR).glob("*.mp3"))

    print(f"Found {len(mp3_files)} MP3 files to process\n")

    for mp3_file in mp3_files:
        # Skip empty files
        if mp3_file.stat().st_size == 0:
            print(f"⚠ Skipping empty file: {mp3_file.name}")
            continue

        # Skip if already has language suffix
        if mp3_file.stem.endswith("-es") or mp3_file.stem.endswith("-en"):
            print(f"✓ Already processed: {mp3_file.name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {mp3_file.name}")
        print(f"Size: {mp3_file.stat().st_size / 1024:.1f} KB")
        print(f"{'=' * 60}")

        # Check if it's a known voice
        base_name = mp3_file.stem.lower()
        detected_lang = None

        # Check known voices
        for known_voice, lang in KNOWN_VOICES.items():
            if known_voice in base_name:
                detected_lang = lang
                print(f"✓ Known voice detected: {known_voice} -> {lang}")
                break

        # If not known, use Whisper
        if not detected_lang:
            print(f"🔍 Detecting language with Whisper...")
            detected_lang = detect_language_whisper(str(mp3_file))

        if detected_lang:
            # Create new filename with language suffix
            new_name = f"{mp3_file.stem}-{detected_lang}{mp3_file.suffix}"
            new_path = mp3_file.parent / new_name

            # Rename file
            mp3_file.rename(new_path)
            print(f"✓ Renamed: {mp3_file.name} -> {new_name}")

            results[mp3_file.stem] = {
                "original": mp3_file.name,
                "new": new_name,
                "language": detected_lang,
            }
        else:
            print(f"⚠ Could not detect language for: {mp3_file.name}")
            results[mp3_file.stem] = {
                "original": mp3_file.name,
                "new": mp3_file.name,
                "language": "unknown",
            }

    return results


def main():
    print("=" * 60)
    print("Voice Language Detection & Renaming")
    print("=" * 60)

    results = process_voice_files()

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    for voice_name, info in results.items():
        lang = info["language"]
        if lang == "es":
            emoji = "🇪🇸"
        elif lang == "en":
            emoji = "🇬🇧"
        else:
            emoji = "❓"

        print(f"{emoji} {info['original']} -> {info['new']} ({lang})")

    # Save results
    results_file = os.path.join(VOICE_SAMPLES_DIR, "voice_detection_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: voice_detection_results.json")


if __name__ == "__main__":
    main()
