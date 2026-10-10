"""
AI Report Service — Ollama (llama3) Edition
Team: CODE CRAFTERS | Hackathon 2K26

Drop-in replacement for gemini_service.py
- Uses Ollama local API instead of Gemini
- Fixes timeout / connection drop after a few minutes
- Keeps the same function signature so dashboard.py needs zero changes
"""

import requests  # type: ignore
import json
import time

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
REQUEST_TIMEOUT = 120        # seconds — increase if your PC is slow
MAX_RETRIES = 3              # retry on connection error
RETRY_DELAY = 2              # seconds between retries

# ── Core function (same name as original) ────────────────────────────────────

def generate_district_report(district: str, disease_data: dict) -> str:
    """
    Generate an AI health report for a district.
    Parameters
    ----------
    district     : str   — district name, e.g. "Chennai"
    disease_data : dict  — aggregated stats from /api/stats or /api/locations
    Returns
    -------
    str — formatted report text, or error message
    """

    prompt = f"""You are a public health analyst. Write a short, clear disease outbreak report for {district} district in Tamil Nadu, India.

Data provided:
{json.dumps(disease_data, indent=2)}

Your report must include:
1. Current disease situation summary
2. High-risk diseases and affected population estimate
3. Recommended immediate actions for health authorities
4. Public safety advice in simple language

Keep the report under 250 words. Be factual and direct."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 400,
            "num_gpu": 50,  # Ensure layers are offloaded to GPU
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response from model.")

        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return (
                "Connection error: Ollama is not running.\n"
                "Fix: open a terminal and run  →  ollama serve"
            )

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return (
                "Timeout: llama3 took too long to respond.\n"
                "Fix: try  ollama pull llama3  to ensure model is fully downloaded,\n"
                "or increase REQUEST_TIMEOUT in gemini_service.py"
            )

        except requests.exceptions.HTTPError as e:
            return f"Ollama API error: {e}"

        except Exception as e:
            return f"Unexpected error: {e}"

    return "Failed after multiple retries. Check if Ollama is running."


# ── Keep-alive ping (call this on app startup) ────────────────────────────────

def ping_ollama() -> bool:
    """Returns True if Ollama is reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Pinging Ollama...", "OK" if ping_ollama() else "FAILED — run: ollama serve")
    sample_data = {
        "district": "Chennai",
        "total_cases": 1240,
        "diseases": {"Dengue": 580, "COVID-19": 420, "Typhoid": 240},
        "risk_level": "HIGH"
    }
    print("\nGenerating report...\n")
    print(generate_district_report("Chennai", sample_data))
