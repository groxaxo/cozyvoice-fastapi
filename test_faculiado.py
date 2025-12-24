#!/usr/bin/env python3
"""Quick test for faculiado-es voice"""

import requests

API_URL = "http://localhost:8000/v1/audio/speech"

data = {
    "model": "cosyvoice3",
    "input": "Hola, soy Facu de Argentina. Esta es una prueba de voz masculina en español.",
    "voice": "faculiado-es",
    "response_format": "wav",
    "speed": 1.0,
}

headers = {"Authorization": "Bearer not-needed", "Content-Type": "application/json"}

print("Testing faculiado-es voice...")
print(f"Voice: {data['voice']}")
print(f"Text: {data['input']}")
print("=" * 60)

try:
    response = requests.post(API_URL, headers=headers, json=data, timeout=60)
    response.raise_for_status()

    with open("test_faculiado_es.wav", "wb") as f:
        f.write(response.content)

    print(f"✓ SUCCESS!")
    print(f"✓ Generated {len(response.content):,} bytes")
    print(f"✓ Saved to: test_faculiado_es.wav")
    print("\nExpected: Spanish male voice (posh, charismatic Argentine)")

except Exception as e:
    print(f"✗ FAILED: {e}")
    if hasattr(e, "response") and e.response is not None:
        print(f"Response: {e.response.text}")
