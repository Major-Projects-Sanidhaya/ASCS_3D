from typing import Optional
from intelligence.ollama_client import is_available, generate, extract_json

MODEL = 'llama3.2:1b'

SYSTEM = """You are the mission intelligence for an autonomous drone swarm.
You receive zone sensor summaries and output tactical JSON decisions.
Respond ONLY with valid JSON, no explanation, no markdown.
JSON structure:
{
  "priority_zone": <int or null>,
  "action": <"SCOUT"|"CONVERGE"|"PERIMETER"|"HOLD"|"WITHDRAW"|"NONE">,
  "target_xy": <[x,y] or null>,
  "reasoning": <string max 15 words>,
  "worker_task": <"NONE"|"INVESTIGATE"|"SUPPRESS"|"EVACUATE">,
  "urgency": <"LOW"|"MEDIUM"|"HIGH">
}"""


class LLMGeneral:
    """
    LLM decision layer for General.
    Called every 15 seconds (configurable).
    Falls back to rule-based logic when Ollama unavailable.
    Never blocks the sim loop — uses short timeout.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled    = enabled
        self._available  = is_available() if enabled else False
        self._last:      Optional[dict] = None
        self._call_count = 0
        self._fail_count = 0
        if self._available:
            print('[LLMGeneral] Ollama connected — mission intelligence active')
        else:
            print('[LLMGeneral] Ollama not available — using rule-based fallback')

    def decide(self, zone_summary: dict, scenario: str) -> dict:
        self._call_count += 1
        if self._enabled and self._available:
            result = self._llm_decide(zone_summary, scenario)
            if result:
                self._last = result
                print(f'[LLMGeneral] {result["action"]} '
                      f'zone={result.get("priority_zone")} '
                      f'urgency={result.get("urgency","?")} '
                      f'→ {result.get("reasoning","")}')
                return result
            self._fail_count += 1
        return self._rule_fallback(zone_summary)

    def _llm_decide(self, zone_summary: dict,
                    scenario: str) -> Optional[dict]:
        lines = [f'Scenario: {scenario}', 'Zone data:']
        for zh, info in zone_summary.items():
            lines.append(
                f'  Zone {zh}: coverage={info["coverage"]:.2f} '
                f'phase={info["op_phase"]} '
                f'authorized={info["worker_auth"]} '
                f'health={info["health"]:.2f}'
            )
        lines.append('Decide the next swarm action.')
        prompt = '\n'.join(lines)

        text = generate(MODEL, prompt, SYSTEM,
                        temperature=0.2, max_tokens=150, timeout=8.0)
        result = extract_json(text)
        if result and self._validate(result):
            return result
        return None

    def _validate(self, d: dict) -> bool:
        valid_actions = {'SCOUT', 'CONVERGE', 'PERIMETER',
                         'HOLD', 'WITHDRAW', 'NONE'}
        return ('action' in d and d.get('action') in valid_actions)

    def _rule_fallback(self, zone_summary: dict) -> dict:
        worst_zh  = None
        worst_cov = 1.0
        for zh, info in zone_summary.items():
            if info.get('coverage', 1.0) < worst_cov:
                worst_cov = info['coverage']
                worst_zh  = zh

        all_tasking = all(
            v.get('op_phase') == 'TASKING'
            for v in zone_summary.values()
        )
        if all_tasking:
            return {'priority_zone': None, 'action': 'HOLD',
                    'target_xy': None, 'reasoning': 'All zones covered',
                    'worker_task': 'NONE', 'urgency': 'LOW'}
        if worst_cov < 0.4:
            return {'priority_zone': worst_zh, 'action': 'SCOUT',
                    'target_xy': None,
                    'reasoning': f'Zone {worst_zh} needs scouting',
                    'worker_task': 'NONE', 'urgency': 'MEDIUM'}
        return {'priority_zone': None, 'action': 'NONE',
                'target_xy': None, 'reasoning': 'Nominal operations',
                'worker_task': 'NONE', 'urgency': 'LOW'}

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_decision(self) -> Optional[dict]:
        return self._last
