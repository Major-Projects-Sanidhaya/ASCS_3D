import os
import sys
import importlib.util
import math
from typing import Optional

BEHAVIOR_DIR = os.path.join(os.path.dirname(__file__), '_generated')


def _ensure_dir():
    os.makedirs(BEHAVIOR_DIR, exist_ok=True)
    init = os.path.join(BEHAVIOR_DIR, '__init__.py')
    if not os.path.exists(init):
        open(init, 'w').close()


def _template_behavior(scout_id: str, zone_hash: int,
                        params: dict) -> str:
    """
    Generates a unique ScoutBehavior subclass from a parameter dict.
    params controls the scout's individual personality:
      w_sep:        separation weight
      w_wp:         waypoint pull weight
      arrival_dist: how close to count as arrived
      max_speed:    top speed
      explore_bias: [dx, dy] bias direction (makes scouts fan out)
    """
    class_name = f'Scout_{scout_id.replace("-", "_")}_Behavior'
    return f'''
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scout.scout_behavior import ScoutBehavior

class {class_name}(ScoutBehavior):
    """
    Auto-generated behavior for scout {scout_id} in zone {zone_hash}.
    Generated at spawn. Destroyed at deletion.
    Personality: w_sep={params["w_sep"]:.3f} w_wp={params["w_wp"]:.3f}
                 max_speed={params["max_speed"]:.2f} bias={params["explore_bias"]}
    """

    MAX_SPEED    = {params["max_speed"]:.4f}
    SEP_RADIUS   = {params.get("sep_radius", 2.0):.4f}
    W_SEP        = {params["w_sep"]:.4f}
    W_WP         = {params["w_wp"]:.4f}
    EXPLORE_BIAS = np.array({params["explore_bias"]})

    def __init__(self, scout_id, zone_hash, patrol_target):
        super().__init__(scout_id, zone_hash, patrol_target)
        self.arrival_dist = {params.get("arrival_dist", 1.2):.4f}

    def compute_velocity(self, pos, vel, neighbor_pos,
                         obs_min, arena_hw, arena_hh):
        v = np.zeros(3)

        # Wall repulsion
        repulse = 2.0
        for axis, half in [(0, arena_hw), (1, arena_hh)]:
            d_hi = half - pos[axis]
            d_lo = pos[axis] + half
            if d_hi < repulse: v[axis] -= (repulse - d_hi) * 4.0
            if d_lo < repulse: v[axis] += (repulse - d_lo) * 4.0

        # Obstacle avoidance
        if obs_min < 1.5 and self._last_pos is not None:
            diff = self._last_pos - pos
            n = float(np.linalg.norm(diff))
            if n > 0.01:
                v += (diff / n) * 3.0

        # Separation from neighbors
        for rp in neighbor_pos:
            dist = float(np.linalg.norm(rp))
            if 0.01 < dist < self.SEP_RADIUS:
                v -= (rp / dist) / (dist + 1e-6) * self.W_SEP

        # Waypoint pull toward patrol target
        wp_vec = self.patrol_target - pos
        wp_dist = float(np.linalg.norm(wp_vec))
        if wp_dist > 0.05:
            v += (wp_vec / wp_dist) * self.W_WP

        # Individual explore bias (unique per scout — causes fan-out)
        v[:2] += self.EXPLORE_BIAS * 0.3

        # Altitude hold
        v[2] = (self.patrol_target[2] - pos[2]) * 2.0

        speed = float(np.linalg.norm(v[:2]))
        if speed > self.MAX_SPEED:
            v[:2] = v[:2] / speed * self.MAX_SPEED
        return v

    def on_destroy(self):
        pass


def create():
    return {class_name}
'''


def _llm_behavior(scout_id: str, zone_hash: int,
                   scout_index: int, total: int,
                   scenario: str) -> Optional[str]:
    """
    Ask Ollama to generate a unique ScoutBehavior subclass.
    Each scout gets a different prompt so behaviors genuinely differ.
    """
    try:
        import urllib.request
        import json

        class_name = f'Scout_{scout_id.replace("-", "_")}_Behavior'

        personalities = [
            'aggressive explorer that prioritizes speed over caution',
            'cautious observer that stays near zone edges and avoids center',
            'systematic sweeper that moves in straight lines',
            'random walker that changes direction frequently',
            'perimeter patrol that circles the zone boundary',
        ]
        personality = personalities[scout_index % len(personalities)]

        prompt = (
            f'Write a Python class named {class_name} for a drone scout.\n'
            f'Scenario: {scenario}\n'
            f'Scout personality: {personality}\n'
            f'Scout index {scout_index} of {total} in zone {zone_hash}.\n\n'
            f'Extend this base structure exactly:\n'
            f'class {class_name}(ScoutBehavior):\n'
            f'    MAX_SPEED = 5.0\n'
            f'    def compute_velocity(self, pos, vel, neighbor_pos, obs_min, arena_hw, arena_hh):\n'
            f'        v = np.zeros(3)\n'
            f'        # YOUR BEHAVIOR HERE (use pos, vel, neighbor_pos, self.patrol_target)\n'
            f'        v[2] = (self.patrol_target[2] - pos[2]) * 2.0\n'
            f'        speed = float(np.linalg.norm(v[:2]))\n'
            f'        if speed > self.MAX_SPEED: v[:2] = v[:2]/speed*self.MAX_SPEED\n'
            f'        return v\n'
            f'    def on_destroy(self): pass\n\n'
            f'Rules:\n'
            f'- numpy is available as np\n'
            f'- Keep under 30 lines\n'
            f'- Must return np.ndarray shape (3,)\n'
            f'- Respond with ONLY the class code'
        )

        payload = json.dumps({
            'model': 'llama3.2:1b',
            'prompt': prompt,
            'system': 'You write Python drone behavior classes. Return only valid Python code. No markdown.',
            'stream': False,
            'options': {'temperature': 0.8, 'num_predict': 500}
        }).encode()
        req = urllib.request.Request(
            'http://localhost:11434/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read()).get('response', '').strip()
            if '```python' in raw:
                raw = raw.split('```python')[1].split('```')[0].strip()
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0].strip()
            if (f'class {class_name}' in raw
                    and 'def compute_velocity' in raw
                    and 'return' in raw):
                return (
                    '\nimport numpy as np\n'
                    'import sys, os\n'
                    'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n'
                    'from Scout.scout_behavior import ScoutBehavior\n\n'
                    f'{raw}\n\n'
                    f'def create():\n'
                    f'    return {class_name}\n'
                )
    except Exception:
        pass
    return None


def generate_behavior(scout_id: str, zone_hash: int,
                       scout_index: int, total_scouts_in_zone: int,
                       use_llm: bool = False,
                       llm_context: str = 'default patrol',
                       scenario: str = 'default') -> type:
    """
    Main entry point. Generates a unique ScoutBehavior subclass,
    saves it to Scout/_generated/, imports and returns the class.

    Each scout gets different parameters based on its index,
    ensuring they fan out rather than cluster.
    """
    _ensure_dir()

    filename    = f'behavior_{scout_id.replace("-", "_")}.py'
    filepath    = os.path.join(BEHAVIOR_DIR, filename)
    module_name = f'Scout._generated.behavior_{scout_id.replace("-", "_")}'

    code = None
    if use_llm:
        print(f'[BehaviorGen] Generating LLM behavior for scout {scout_id}...')
        code = _llm_behavior(scout_id, zone_hash,
                             scout_index, total_scouts_in_zone, scenario)
        if code:
            print(f'[BehaviorGen] LLM behavior generated for {scout_id}')
        else:
            print(f'[BehaviorGen] LLM failed — using template for {scout_id}')

    if code is None:
        import random
        rng = random.Random(hash(scout_id) % (2 ** 32))
        n = max(total_scouts_in_zone, 1)
        angle = (2 * math.pi * scout_index / n) + rng.uniform(-0.3, 0.3)
        bias_mag = rng.uniform(0.4, 0.9)
        bias = [round(math.cos(angle) * bias_mag, 4),
                round(math.sin(angle) * bias_mag, 4)]
        params = {
            'w_sep':        rng.uniform(0.7, 1.2),
            'w_wp':         rng.uniform(2.0, 3.0),
            'max_speed':    rng.uniform(4.0, 6.0),
            'sep_radius':   rng.uniform(1.5, 2.5),
            'arrival_dist': rng.uniform(0.8, 1.5),
            'explore_bias': bias,
        }
        code = _template_behavior(scout_id, zone_hash, params)

    with open(filepath, 'w') as f:
        f.write(code)

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec   = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        behavior_class = module.create()
        return behavior_class
    except Exception as e:
        print(f'[BehaviorGen] Generated code failed ({e}) — using default')
        from Scout.scout_behavior import DefaultScoutBehavior
        return DefaultScoutBehavior


def destroy_behavior(scout_id: str) -> None:
    """
    Deletes the generated behavior file for this scout.
    Called when scout is removed from the swarm.
    """
    filename    = f'behavior_{scout_id.replace("-", "_")}.py'
    filepath    = os.path.join(BEHAVIOR_DIR, filename)
    module_name = f'Scout._generated.behavior_{scout_id.replace("-", "_")}'

    if module_name in sys.modules:
        del sys.modules[module_name]

    if os.path.exists(filepath):
        os.remove(filepath)

    pyc = filepath + 'c'
    if os.path.exists(pyc):
        os.remove(pyc)
