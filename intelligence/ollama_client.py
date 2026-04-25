import json
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_BASE = 'http://localhost:11434'


def is_available(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f'{OLLAMA_BASE}/api/tags')
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def generate(model: str, prompt: str, system: str,
             temperature: float = 0.2,
             max_tokens: int = 200,
             timeout: float = 10.0) -> Optional[str]:
    """
    Synchronous call to Ollama /api/generate.
    Returns response text or None on failure.
    Non-blocking timeout prevents stalling the sim loop.
    """
    payload = json.dumps({
        'model':  model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        },
    }).encode()
    try:
        req = urllib.request.Request(
            f'{OLLAMA_BASE}/api/generate',
            data    = payload,
            headers = {'Content-Type': 'application/json'},
            method  = 'POST',
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()).get('response', '').strip()
    except Exception:
        return None


def extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from a text response."""
    if not text:
        return None
    start = text.find('{')
    end   = text.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    return None
