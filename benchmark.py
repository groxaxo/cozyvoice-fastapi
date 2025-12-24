#!/usr/bin/env python3
"""
Benchmark script for CosyVoice FastAPI server
Measures: tokens/second, latency, throughput
"""

import time
import requests
import json
import sys
from typing import List, Dict
import statistics

BASE_URL = "http://localhost:8000"

# Test sentences of varying lengths
TEST_SENTENCES = {
    "short": "Hello world.",
    "medium": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
    "long": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。In this modern era of technology, we find ourselves constantly adapting to new innovations.",
    "very_long": "The advancement of artificial intelligence and machine learning has revolutionized the way we interact with technology. From voice assistants to autonomous vehicles, these technologies are reshaping our daily lives in unprecedented ways. Natural language processing, in particular, has made significant strides in recent years, enabling machines to understand and generate human-like text with remarkable accuracy.",
}

VOICES = ["en", "es", "lucho-es", "aimee-en"]


def benchmark_request(text: str, voice: str, runs: int = 3) -> Dict:
    """Run a single benchmark test"""
    latencies = []

    for i in range(runs):
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "model": "cosyvoice3",
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
            timeout=60,
        )

        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None

        # Get audio size
        audio_size = len(response.content)

    # Calculate statistics
    char_count = len(text)
    word_count = len(text.split())
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # Calculate tokens/second (approximation: characters as proxy for tokens)
    tokens_per_second = char_count / avg_latency
    words_per_second = word_count / avg_latency

    return {
        "text_length": char_count,
        "word_count": word_count,
        "voice": voice,
        "avg_latency_s": round(avg_latency, 3),
        "min_latency_s": round(min_latency, 3),
        "max_latency_s": round(max_latency, 3),
        "chars_per_second": round(tokens_per_second, 2),
        "words_per_second": round(words_per_second, 2),
        "audio_size_kb": round(audio_size / 1024, 2),
        "runs": runs,
    }


def print_results(results: List[Dict]):
    """Print benchmark results in a nice table"""
    print("\n" + "=" * 80)
    print("COSYVOICE BENCHMARK RESULTS")
    print("=" * 80)

    for result in results:
        print(
            f"\nText Length: {result['text_length']} chars | Words: {result['word_count']} | Voice: {result['voice']}"
        )
        print(
            f"  Avg Latency: {result['avg_latency_s']}s | Min: {result['min_latency_s']}s | Max: {result['max_latency_s']}s"
        )
        print(
            f"  ⚡ Chars/sec: {result['chars_per_second']} | Words/sec: {result['words_per_second']}"
        )
        print(f"  Audio Size: {result['audio_size_kb']} KB")
        print(f"  RTF (Real-Time Factor): {calculate_rtf(result):.2f}x")


def calculate_rtf(result: Dict) -> float:
    """Calculate Real-Time Factor (RTF)
    RTF = generation_time / audio_duration
    Lower is better (e.g., 0.1 means 10x faster than real-time)
    """
    # Assume 25kHz sample rate, mono, 16-bit (2 bytes) from WAV
    # Audio duration ≈ (audio_size_bytes - 44_header) / (sample_rate * 2)
    sample_rate = 25000  # CosyVoice3 uses 25kHz
    audio_bytes = result["audio_size_kb"] * 1024 - 44  # Remove WAV header
    audio_duration_s = audio_bytes / (sample_rate * 2)  # 2 bytes per sample

    rtf = result["avg_latency_s"] / audio_duration_s if audio_duration_s > 0 else 0
    return rtf


def main():
    print("🚀 Starting CosyVoice Benchmark...")
    print(f"📡 Server: {BASE_URL}")

    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        sys.exit(1)

    results = []

    # Benchmark different text lengths with different voices
    for length_name, text in TEST_SENTENCES.items():
        print(f"\n📝 Testing {length_name.upper()} text ({len(text)} chars)...")

        # Test with one voice for each text length
        voice = VOICES[0]  # Use English voice for consistency
        result = benchmark_request(text, voice, runs=3)

        if result:
            results.append(result)
            print(
                f"   ✓ Completed: {result['avg_latency_s']}s | {result['chars_per_second']} chars/s"
            )

    # Benchmark different voices with medium text
    print(f"\n🎭 Testing different VOICES with medium text...")
    medium_text = TEST_SENTENCES["medium"]

    for voice in VOICES[1:]:  # Skip first one (already tested)
        result = benchmark_request(medium_text, voice, runs=2)
        if result:
            results.append(result)
            print(
                f"   ✓ {voice}: {result['avg_latency_s']}s | {result['chars_per_second']} chars/s"
            )

    # Print final results
    print_results(results)

    # Summary statistics
    all_chars_per_sec = [r["chars_per_second"] for r in results]
    all_latencies = [r["avg_latency_s"] for r in results]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average Characters/Second: {statistics.mean(all_chars_per_sec):.2f}")
    print(f"Average Latency: {statistics.mean(all_latencies):.3f}s")
    print(
        f"Min/Max Chars/Second: {min(all_chars_per_sec):.2f} / {max(all_chars_per_sec):.2f}"
    )
    print(f"Total Tests Run: {len(results)}")
    print("=" * 80 + "\n")

    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "server": BASE_URL,
                "results": results,
                "summary": {
                    "avg_chars_per_second": round(
                        statistics.mean(all_chars_per_sec), 2
                    ),
                    "avg_latency_s": round(statistics.mean(all_latencies), 3),
                    "total_tests": len(results),
                },
            },
            f,
            indent=2,
        )

    print("💾 Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
