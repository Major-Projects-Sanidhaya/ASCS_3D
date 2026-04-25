from typing import Optional
from intelligence.ollama_client import is_available, generate, extract_json

MODEL = 'llama3.2:1b'

SYSTEM = """You are the tactical intelligence for one drone squad leader.
You decide how to adjust the squad's behavior based on sensor data.
Respond ONLY with valid JSON, no explanation.
JSON structure:
{
  "adjust_sep": <float -0.25 to 0.25>,
  "adjust_align": <float -0.25 to 0.25>,
  "adjust_coh": <float -0.25 to 0.25>,
  "patrol_mode": <"SCOUT"|"HOLD"|"CONVERGE"|null>,
  "reasoning": <string max 10 words>
}"""


class LLMNode:
    """
    Per-Node LLM tactical layer.
    Called every 30 seconds to fine-tune Reynolds weights.
    Much lighter than General's LLM — focuses only on this zone.
    Each Node instance has its own LLMNode. They operate independently.
    No cross-node communication — anonymity preserved.
    """

    def __init__(self, zone_hash: int, enabled: bool = True) -> None:
        self._zone_hash = zone_hash
        self._enabled   = enabled
        self._available = is_available() if enabled else False
        self._last:     Optional[dict] = None

    def advise(self, zone_hash: int, coverage: float, scout_count: int,
               mean_speed: float, op_phase: str, frames_scouting: int,
               scenario: str) -> dict:
        """Returns Reynolds weight adjustments for this Node."""
        if self._enabled and self._available:
            result = self._llm_advise(
                zone_hash, coverage, scout_count,
                mean_speed, op_phase, frames_scouting, scenario)
            if result:
                self._last = result
                return result

        return self._rule_fallback(coverage, op_phase)

    def _llm_advise(self, zone_hash: int, coverage: float, scout_count: int,
                    mean_speed: float, op_phase: str, frames_scouting: int,
                    scenario: str) -> Optional[dict]:
        prompt = (
            f'Scenario: {scenario}\n'
            f'Zone {zone_hash}: coverage={coverage:.2f} '
            f'scouts={scout_count} '
            f'mean_speed={mean_speed:.2f}m/s '
            f'phase={op_phase} '
            f'frames_scouting={frames_scouting}\n'
            f'Advise Reynolds weight adjustments.'
        )
        text = generate(MODEL, prompt, SYSTEM,
                        temperature=0.3, max_tokens=100, timeout=6.0)
        result = extract_json(text)
        if result and all(k in result for k in
                          ('adjust_sep', 'adjust_align', 'adjust_coh')):
            for k in ('adjust_sep', 'adjust_align', 'adjust_coh'):
                result[k] = float(max(-0.25, min(0.25, result.get(k, 0.0))))
            return result
        return None

    def _rule_fallback(self, coverage: float, op_phase: str) -> dict:
        if op_phase == 'SCOUTING' and coverage < 0.3:
            return {'adjust_sep': 0.15, 'adjust_align': -0.05,
                    'adjust_coh': -0.1, 'patrol_mode': None,
                    'reasoning': 'Low coverage — increase separation'}
        if op_phase == 'TASKING':
            return {'adjust_sep': -0.05, 'adjust_align': 0.1,
                    'adjust_coh': 0.1, 'patrol_mode': None,
                    'reasoning': 'Tasking — tighten formation'}
        return {'adjust_sep': 0.0, 'adjust_align': 0.0,
                'adjust_coh': 0.0, 'patrol_mode': None,
                'reasoning': 'Nominal'}

    @property
    def last_advice(self) -> Optional[dict]:
        return self._last
