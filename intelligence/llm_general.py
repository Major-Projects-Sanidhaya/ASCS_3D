import json
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_URL    = 'http://localhost:11434/api/generate'
DEFAULT_MODEL = 'llama3.2:1b'

SYSTEM_PROMPT = """You are the intelligence layer of an autonomous drone swarm
coordinating a fire response mission. You receive zone sensor data and output
tactical decisions in JSON. You must respond ONLY with valid JSON, no explanation.

Your response must always be this exact structure:
{
  "priority_zone": <int or null>,
  "action": <"SCOUT"|"CONVERGE"|"PERIMETER"|"HOLD"|"WITHDRAW"|"NONE">,
  "target": <[x, y] or null>,
  "reasoning": <string, max 20 words>,
  "worker_task": <"NONE"|"INVESTIGATE"|"SUPPRESS"|"EVACUATE">
}"""


class LLMGeneral:
    """
    LLM-powered decision layer for the General.
    Reads zone summaries and outputs tactical decisions.
    Falls back to rule-based logic if LLM is unavailable.
    Non-blocking: uses a simple HTTP call with timeout.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 enabled: bool = True) -> None:
        self._model   = model
        self._enabled = enabled
        self._last_decision: Optional[dict] = None
        self._call_count  = 0
        self._fail_count  = 0
        self._available   = False
        if enabled:
            self._available = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            req = urllib.request.Request('http://localhost:11434/api/tags')
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            print('[LLM] Ollama not running — using rule-based fallback')
            print('[LLM] Start with: ollama serve')
            return False

    def _call(self, prompt: str) -> Optional[dict]:
        if not self._available:
            return None
        payload = json.dumps({
            'model':  self._model,
            'prompt': prompt,
            'system': SYSTEM_PROMPT,
            'stream': False,
            'options': {'temperature': 0.2, 'num_predict': 150}
        }).encode()
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read())
                raw = result.get('response', '').strip()
                start = raw.find('{')
                end   = raw.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
        except Exception as e:
            self._fail_count += 1
            if self._fail_count <= 3:
                print(f'[LLM] Call failed: {e}')
        return None

    def decide(self, zone_summary: dict,
               scenario: str = 'default') -> dict:
        """
        Main decision method. Called by General every emit_interval.
        Returns a decision dict General can act on.
        """
        self._call_count += 1

        if self._enabled and self._available:
            prompt = self._build_prompt(zone_summary, scenario)
            decision = self._call(prompt)
            if decision and self._validate(decision):
                self._last_decision = decision
                print(f'[LLM] Decision: {decision["action"]} '
                      f'zone={decision["priority_zone"]} '
                      f'→ {decision["reasoning"]}')
                return decision

        return self._rule_based_fallback(zone_summary)

    def _build_prompt(self, zone_summary: dict,
                      scenario: str) -> str:
        lines = [f'Scenario: {scenario}', 'Zone sensor data:']
        for zh, info in zone_summary.items():
            lines.append(
                f'  Zone {zh}: coverage={info["coverage"]:.2f} '
                f'phase={info["op_phase"]} '
                f'auth={info["worker_auth"]} '
                f'health={info["health"]:.2f}'
            )
        lines.append('What tactical action should the swarm take next?')
        return '\n'.join(lines)

    def _validate(self, d: dict) -> bool:
        required = ['action', 'priority_zone', 'reasoning']
        valid_actions = {'SCOUT', 'CONVERGE', 'PERIMETER',
                         'HOLD', 'WITHDRAW', 'NONE'}
        return (all(k in d for k in required)
                and d.get('action') in valid_actions)

    def _rule_based_fallback(self, zone_summary: dict) -> dict:
        """Simple rule-based decisions when LLM unavailable."""
        worst = min(zone_summary.items(),
                    key=lambda x: x[1].get('coverage', 1.0),
                    default=(None, {}))
        if worst[0] is not None and worst[1].get('coverage', 1.0) < 0.5:
            return {'priority_zone': worst[0], 'action': 'SCOUT',
                    'target': None, 'reasoning': 'Low coverage zone needs scouting',
                    'worker_task': 'NONE'}
        all_tasking = all(
            v.get('op_phase') == 'TASKING'
            for v in zone_summary.values()
        )
        if all_tasking:
            return {'priority_zone': None, 'action': 'HOLD',
                    'target': None, 'reasoning': 'All zones covered, holding',
                    'worker_task': 'NONE'}
        return {'priority_zone': None, 'action': 'NONE',
                'target': None, 'reasoning': 'Nominal operations',
                'worker_task': 'NONE'}

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_decision(self) -> Optional[dict]:
        return self._last_decision
