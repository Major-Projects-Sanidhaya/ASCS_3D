import argparse
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from typing import Dict, List, Optional
from controllers.swarm_controller import SwarmController
from logger.movement_logger import MovementLogger

# ── Config ────────────────────────────────────────────────────────────────────

def build_config() -> dict:
    parser = argparse.ArgumentParser(
        description='ASCS_3D — Anonymous Drone Swarm\n'
                    'Use cases: house fire, forest fire, search & rescue\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--arena-w',   type=float, default=20.0,
        help='Arena width  in metres (default: 20)')
    parser.add_argument('--arena-h',   type=float, default=20.0,
        help='Arena height in metres (default: 20)')
    parser.add_argument('--grid-cols', type=int,   default=2,
        help='Zone grid columns (default: 2 → 4 nodes for 20m arena)')
    parser.add_argument('--grid-rows', type=int,   default=2,
        help='Zone grid rows    (default: 2)')
    parser.add_argument('--scouts',    type=int,   default=4,
        help='Scouts per node   (default: 4)')
    parser.add_argument('--workers',   type=int,   default=1,
        help='Workers per node  (default: 1)')
    parser.add_argument('--altitude',  type=float, default=3.0,
        help='Flight altitude   (default: 3.0m)')
    parser.add_argument('--emit-interval', type=float, default=5.0,
        help='Goal token emit interval in seconds (default: 5)')
    parser.add_argument('--scenario',  type=str,   default='default',
        choices=['default', 'house_fire', 'forest_fire', 'search_rescue'],
        help='Mission scenario preset')
    parser.add_argument('--llm-scouts', action='store_true',
        help='Generate scout behaviors via Ollama LLM (requires ollama serve)')
    parser.add_argument('--physics', default='simple',
        choices=['simple', 'pyflyt'],
        help='Physics backend for drone flight dynamics (default: simple)')
    args = parser.parse_args()

    presets = {
        'house_fire': {
            'arena_w': 15.0, 'arena_h': 15.0,
            'grid_cols': 2,  'grid_rows': 2,
            'n_scouts_per_node': 3, 'n_workers_per_node': 2,
            'altitude': 2.0,
            'emit_interval': 3.0
        },
        'forest_fire': {
            'arena_w': 100.0, 'arena_h': 100.0,
            'grid_cols': 4,   'grid_rows': 4,
            'n_scouts_per_node': 6, 'n_workers_per_node': 2,
            'altitude': 8.0,
            'emit_interval': 8.0
        },
        'search_rescue': {
            'arena_w': 50.0, 'arena_h': 50.0,
            'grid_cols': 3,  'grid_rows': 3,
            'n_scouts_per_node': 5, 'n_workers_per_node': 2,
            'altitude': 5.0,
            'emit_interval': 6.0
        },
    }

    if args.scenario != 'default' and args.scenario in presets:
        cfg = presets[args.scenario]
        print(f'[Config] Scenario: {args.scenario}')
    else:
        cfg = {
            'arena_w':            args.arena_w,
            'arena_h':            args.arena_h,
            'grid_cols':          args.grid_cols,
            'grid_rows':          args.grid_rows,
            'n_scouts_per_node':  args.scouts,
            'n_workers_per_node': args.workers,
            'altitude':           args.altitude,
            'emit_interval':      args.emit_interval,
        }

    max_zone_w = cfg['arena_w'] / cfg['grid_cols']
    max_zone_h = cfg['arena_h'] / cfg['grid_rows']
    if max_zone_w > 35 or max_zone_h > 35:
        suggested_cols = max(2, math.ceil(cfg['arena_w'] / 30))
        suggested_rows = max(2, math.ceil(cfg['arena_h'] / 30))
        print(f'[Config] WARNING: Zone size {max_zone_w:.0f}x{max_zone_h:.0f}m '
              f'exceeds UWB range. Suggest --grid-cols {suggested_cols} '
              f'--grid-rows {suggested_rows}')

    n_nodes   = cfg['grid_cols'] * cfg['grid_rows']
    n_scouts  = n_nodes * cfg['n_scouts_per_node']
    n_workers = n_nodes * cfg['n_workers_per_node']
    print(f'[Config] Arena: {cfg["arena_w"]}x{cfg["arena_h"]}m  '
          f'Zones: {n_nodes}  Scouts: {n_scouts}  Workers: {n_workers}  '
          f'Alt: {cfg["altitude"]}m')
    cfg['scenario']        = args.scenario
    cfg['use_llm_scouts']  = args.llm_scouts
    cfg['physics']         = args.physics
    return cfg

SIM_HZ   = 60
COLORS   = {
    'general': [0.45, 0.20, 0.80, 1],
    'nodes':   [0.10, 0.62, 0.52, 1],
    'scouts':  [0.94, 0.62, 0.15, 1],
    'workers': [0.22, 0.45, 0.88, 1],
}
RADII    = {'general': 0.35, 'nodes': 0.22, 'scouts': 0.15, 'workers': 0.18}

# ── PyBullet helpers ──────────────────────────────────────────────────────────

def pb_safe(fn, *args, **kwargs):
    """Call a PyBullet function, silently ignore if not connected."""
    try:
        if p.isConnected():
            return fn(*args, **kwargs)
    except Exception:
        pass
    return None


def setup_pybullet() -> None:
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.loadURDF('plane.urdf')
    p.resetDebugVisualizerCamera(
        cameraDistance=28, cameraYaw=35,
        cameraPitch=-30, cameraTargetPosition=[0, 0, 3]
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)


def create_drone_bodies(swarm: SwarmController) -> Dict[str, List[int]]:
    bodies: Dict[str, List[int]] = {}
    for tier, positions in swarm.get_all_positions().items():
        bodies[tier] = []
        r = RADII[tier]
        for pos in positions:
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=r, rgbaColor=COLORS[tier])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            bid = p.createMultiBody(0, col, vis, pos.tolist())
            bodies[tier].append(bid)
    return bodies


def update_drone_bodies(bodies: Dict[str, List[int]],
                        swarm: SwarmController) -> None:
    for tier, positions in swarm.get_all_positions().items():
        for bid, pos in zip(bodies.get(tier, []), positions):
            pb_safe(p.resetBasePositionAndOrientation,
                    bid, pos.tolist(), [0, 0, 0, 1])


def setup_sliders() -> Dict[str, int]:
    return {
        'w_sep':   p.addUserDebugParameter('w_sep   (separation)',   0.0, 2.0, 0.6),
        'w_align': p.addUserDebugParameter('w_align (alignment)',    0.0, 2.0, 0.4),
        'w_coh':   p.addUserDebugParameter('w_coh   (cohesion)',     0.0, 2.0, 0.5),
        'w_wp':    p.addUserDebugParameter('w_wp    (waypoint pull)', 0.0, 1.5, 0.6),
    }


def read_sliders(sliders: Dict[str, int]) -> Dict[str, float]:
    out = {}
    for k, sid in sliders.items():
        try:
            out[k] = float(p.readUserDebugParameter(sid))
        except Exception:
            out[k] = 0.5
    return out


def add_arena_markers(swarm: SwarmController) -> None:
    """Draw zone grid on the ground and waypoint discs."""
    zm = swarm._zone_map
    hw, hh = zm.arena_w / 2, zm.arena_h / 2

    # Arena boundary
    corners = [[-hw,-hh,0],[ hw,-hh,0],[ hw, hh,0],[-hw, hh,0],[-hw,-hh,0]]
    for i in range(len(corners)-1):
        pb_safe(p.addUserDebugLine,
                corners[i], corners[i+1], [0.3,0.3,0.3], 0.5)

    # Zone grid lines
    cols, rows = zm.grid_cols, zm.grid_rows
    cw, ch = zm.arena_w / cols, zm.arena_h / rows
    for i in range(1, cols):
        x = -hw + i * cw
        pb_safe(p.addUserDebugLine, [x,-hh,0],[x, hh,0],[0.2,0.2,0.3],0.4)
    for i in range(1, rows):
        y = -hh + i * ch
        pb_safe(p.addUserDebugLine, [-hw,y,0],[ hw,y,0],[0.2,0.2,0.3],0.4)

    # Waypoint discs
    for wp in swarm._general._agent._waypoint_sequence:
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.4, length=0.04,
            rgbaColor=[0.3, 0.3, 0.3, 0.35])
        p.createMultiBody(0, -1, vis, [wp[0], wp[1], 0.02])


# ── HUD text management ───────────────────────────────────────────────────────

class HUD:
    """
    Manages all floating PyBullet debug text.
    Uses replace so we never accumulate thousands of text objects.
    All draw calls go through pb_safe so disconnect never crashes.
    """

    # Fixed positions for persistent HUD lines
    HUD_MAIN   = [0, 0, 7.0]
    HUD_MODE   = [0, 0, 6.4]
    HUD_KEYS   = [0, 0, 5.8]

    def __init__(self):
        self._ids: Dict[str, int] = {}   # key -> debug item UID
        self._agent_ids: Dict[str, int] = {}  # agent_key -> UID
        self._mode_idx  = 0
        self.MODES      = ['OFF', 'VELOCITY', 'WEIGHTS', 'STATE', 'OP_PHASE']

    @property
    def mode(self) -> str:
        return self.MODES[self._mode_idx]

    def cycle_mode(self) -> None:
        self._mode_idx = (self._mode_idx + 1) % len(self.MODES)
        # Clear all agent labels when mode changes
        for uid in self._agent_ids.values():
            pb_safe(p.removeUserDebugItem, uid)
        self._agent_ids.clear()

    def _set(self, key: str, text: str, pos: list,
             color: list, size: float = 1.0) -> None:
        """Add or replace a named text item."""
        if not p.isConnected():
            return
        try:
            if key in self._ids:
                p.addUserDebugText(
                    text, pos, textColorRGB=color, textSize=size,
                    replaceItemUniqueId=self._ids[key])
            else:
                uid = p.addUserDebugText(
                    text, pos, textColorRGB=color, textSize=size)
                if uid is not None and uid >= 0:
                    self._ids[key] = uid
        except Exception:
            self._ids.pop(key, None)

    def update_main(self, status: dict, manual: bool,
                    bias: np.ndarray) -> None:
        ph = status.get('mission_phase', '?')
        az = status.get('active_zones', '?')
        h  = status.get('overall_health', 1.0)
        mode_str = 'MANUAL' if manual else 'AUTO'
        self._set('main',
            f'Phase:{ph}  Zones:{az}  Health:{h:.2f}  {mode_str}',
            self.HUD_MAIN, [0.9, 0.9, 0.9], 1.1)
        self._set('mode_lbl',
            f'Debug overlay: {self.mode}  (TAB to cycle)',
            self.HUD_MODE, [0.6, 0.6, 0.8], 0.9)
        ctrl = ('MANUAL: W/S=fwd/bk  A/D=left/right  Q/E=alt  R=reset  '
                'SPACE=auto') if manual else (
                'C=converge  P=perimeter  U=auto  '
                '==add_scout  -=remove_scout  SPACE=manual  TAB=debug')
        self._set('keys', ctrl, self.HUD_KEYS, [0.5, 0.5, 0.5], 0.8)

    def update_agent_labels(self, swarm: SwarmController) -> None:
        if self.mode == 'OFF':
            return

        agents = []
        agents.append(('gen_0', swarm._general._agent, 'GENERAL',
                        [0.8, 0.6, 1.0]))
        for i, nc in enumerate(swarm._nodes.values()):
            agents.append((f'node_{i}', nc._agent, 'NODE',
                           [0.4, 1.0, 0.8]))
        for i, sc in enumerate(swarm._scouts):
            agents.append((f'scout_{i}', sc._agent, 'SCOUT',
                           [1.0, 0.9, 0.4]))
        for i, wc in enumerate(swarm._workers):
            agents.append((f'worker_{i}', wc._agent, 'WORKER',
                           [0.5, 0.7, 1.0]))

        for key, agent, tier, color in agents:
            text = self._agent_label(agent, tier)
            if not text:
                continue
            pos = agent.pos.tolist()
            pos[2] += 0.55
            try:
                if key in self._agent_ids:
                    p.addUserDebugText(
                        text, pos, textColorRGB=color, textSize=0.65,
                        replaceItemUniqueId=self._agent_ids[key])
                else:
                    uid = p.addUserDebugText(
                        text, pos, textColorRGB=color, textSize=0.65)
                    if uid is not None and uid >= 0:
                        self._agent_ids[key] = uid
            except Exception:
                self._agent_ids.pop(key, None)

    def _agent_label(self, agent, tier: str) -> str:
        m = self.mode
        if m == 'VELOCITY':
            spd = float(np.linalg.norm(agent.vel))
            vx, vy = agent.vel[0], agent.vel[1]
            d = ''
            if   vx >  0.15: d += '+X'
            elif vx < -0.15: d += '-X'
            if   vy >  0.15: d += '+Y'
            elif vy < -0.15: d += '-Y'
            return f'{spd:.2f}m/s {d or "."}'
        elif m == 'WEIGHTS':
            if tier == 'NODE':
                ws, wa, wc = agent.get_effective_weights()
                return f's={ws:.2f} a={wa:.2f} c={wc:.2f}'
            elif tier == 'SCOUT':
                return f'cmd={agent._cmd_age:.1f}s'
            elif tier == 'WORKER':
                return agent._task_state
            elif tier == 'GENERAL':
                return agent._mission_phase
        elif m == 'STATE':
            if tier == 'GENERAL':
                return agent._mission_phase
            elif tier == 'NODE':
                return f'{agent._mode} t={agent._token_age:.1f}s'
            elif tier == 'SCOUT':
                ok = agent._cmd_age < 0.6
                return f'{"OK" if ok else "LOITER"}'
            elif tier == 'WORKER':
                return agent._task_state
        elif m == 'OP_PHASE':
            if tier == 'NODE':
                op  = agent.get_op_phase() if hasattr(agent,'get_op_phase') else '?'
                cov = getattr(agent, '_coverage_fraction', 0.0)
                return f'{op} {cov:.2f}'
            elif tier == 'SCOUT':
                return f'seq={agent._seq}'
            elif tier == 'WORKER':
                return agent._task_state
            elif tier == 'GENERAL':
                return agent._mission_phase
        return ''


# ── Directional controller ────────────────────────────────────────────────────

class DirectionController:
    """
    Reads PyBullet keyboard events.
    SPACE  -> toggle manual / autonomous
    W/S    -> forward / backward  (+X / -X)
    A/D    -> left / right        (-Y / +Y)
    Q/E    -> altitude up / down
    R      -> reset bias to zero
    TAB    -> cycle HUD overlay mode
    """

    KEY_W     = ord('w');  KEY_S = ord('s')
    KEY_A     = ord('a');  KEY_D = ord('d')
    KEY_Q     = ord('q');  KEY_E = ord('e')
    KEY_R     = ord('r')
    KEY_SPACE = ord(' ')
    KEY_TAB   = 9
    KEY_C     = ord('c')
    KEY_P     = ord('p')
    KEY_U     = ord('u')
    KEY_PLUS  = ord('=')
    KEY_MINUS = ord('-')
    STEP      = 0.12
    ALT_STEP  = 0.05

    def __init__(self, arena_w: float, arena_h: float,
                 default_alt: float):
        self._arena_w     = arena_w
        self._arena_h     = arena_h
        self._bias        = np.zeros(2)
        self._alt         = default_alt
        self._default_alt = default_alt
        self._manual      = False
        self._add_zone_idx   = 0
        self.pending_spawns  = []
        self.pending_despawns = []

    @property
    def manual(self) -> bool:
        return self._manual

    def process(self, swarm: SwarmController,
                hud: HUD) -> None:
        """Read keys and apply bias to Node waypoints."""
        try:
            keys = p.getKeyboardEvents()
        except Exception:
            return

        # Toggle manual
        if (self.KEY_SPACE in keys
                and keys[self.KEY_SPACE] & p.KEY_WAS_TRIGGERED):
            self._manual = not self._manual
            if not self._manual:
                self._bias = np.zeros(2)

        # Cycle debug overlay
        if (self.KEY_TAB in keys
                and keys[self.KEY_TAB] & p.KEY_WAS_TRIGGERED):
            hud.cycle_mode()

        # Reset
        if (self.KEY_R in keys
                and keys[self.KEY_R] & p.KEY_WAS_TRIGGERED):
            self._bias  = np.zeros(2)
            self._alt   = self._default_alt

        # Collective converge to arena centre
        if (self.KEY_C in keys
                and keys[self.KEY_C] & p.KEY_WAS_TRIGGERED):
            centre = np.array([0.0, 0.0, swarm._altitude])
            swarm._general._agent.command_collective_converge(
                centre, [n._agent for n in swarm._nodes.values()])

        # Collective perimeter around detected hotspot
        if (self.KEY_P in keys
                and keys[self.KEY_P] & p.KEY_WAS_TRIGGERED):
            hotspot = swarm._general._agent.detect_hotspot()
            if hotspot:
                c = hotspot['centroid']
                centre = np.array([c[0], c[1], swarm._altitude])
                swarm._general._agent.command_collective_perimeter(
                    centre, 5.0,
                    [n._agent for n in swarm._nodes.values()])
            else:
                print('[Manual] No hotspot detected')

        # Return all nodes to autonomous operation
        if (self.KEY_U in keys
                and keys[self.KEY_U] & p.KEY_WAS_TRIGGERED):
            swarm._general._agent.command_return_autonomous(
                [n._agent for n in swarm._nodes.values()])

        # Add a scout to the next zone (cycling)
        if (self.KEY_PLUS in keys
                and keys[self.KEY_PLUS] & p.KEY_WAS_TRIGGERED):
            zones = list(swarm._nodes.keys())
            if zones:
                zh    = zones[self._add_zone_idx % len(zones)]
                new_sc = swarm.add_scout(zh)
                if new_sc:
                    self.pending_spawns.append(('scouts', new_sc))
                self._add_zone_idx += 1

        # Remove the most recently added scout
        if (self.KEY_MINUS in keys
                and keys[self.KEY_MINUS] & p.KEY_WAS_TRIGGERED):
            if swarm._scouts:
                sc = swarm._scouts[-1]
                swarm.remove_scout(sc)
                self.pending_despawns.append(sc)

        if self._manual:
            hw = self._arena_w / 2 - 1.5
            hh = self._arena_h / 2 - 1.5

            if self.KEY_W in keys and keys[self.KEY_W] & p.KEY_IS_DOWN:
                self._bias[0] = min(self._bias[0] + self.STEP,  hw)
            if self.KEY_S in keys and keys[self.KEY_S] & p.KEY_IS_DOWN:
                self._bias[0] = max(self._bias[0] - self.STEP, -hw)
            if self.KEY_D in keys and keys[self.KEY_D] & p.KEY_IS_DOWN:
                self._bias[1] = min(self._bias[1] + self.STEP,  hh)
            if self.KEY_A in keys and keys[self.KEY_A] & p.KEY_IS_DOWN:
                self._bias[1] = max(self._bias[1] - self.STEP, -hh)
            if self.KEY_Q in keys and keys[self.KEY_Q] & p.KEY_IS_DOWN:
                self._alt = min(self._alt + self.ALT_STEP, 8.0)
            if self.KEY_E in keys and keys[self.KEY_E] & p.KEY_IS_DOWN:
                self._alt = max(self._alt - self.ALT_STEP, 1.5)

            # Push bias into every Node waypoint
            for node in swarm._nodes.values():
                centre = swarm._zone_map.get_zone_centre(
                    node._agent.zone_hash)
                node._agent._waypoint = np.array([
                    centre[0] + self._bias[0],
                    centre[1] + self._bias[1],
                    self._alt,
                ])


# ── Hierarchy connection lines ────────────────────────────────────────────────

def draw_hierarchy_lines(swarm: SwarmController) -> None:
    """Draw node->scout (teal) and node->worker (blue) lines each frame."""
    if not p.isConnected():
        return
    lifetime = (1.0 / SIM_HZ) * 4
    try:
        for nc in swarm._nodes.values():
            npos = nc._agent.pos.tolist()
            for sc in nc._scout_controllers:
                p.addUserDebugLine(
                    npos, sc._agent.pos.tolist(),
                    lineColorRGB=[0.1, 0.6, 0.5],
                    lineWidth=0.5, lifeTime=lifetime)
            for wc in nc._worker_controllers:
                p.addUserDebugLine(
                    npos, wc._agent.pos.tolist(),
                    lineColorRGB=[0.2, 0.4, 0.9],
                    lineWidth=0.6, lifeTime=lifetime)
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_pybullet()

    cfg    = build_config()
    swarm  = SwarmController(cfg)
    swarm._general._agent._scenario = cfg.get('scenario', 'default')

    physics_backend = cfg.get('physics', 'simple')
    if physics_backend == 'pyflyt':
        print('[Physics] Using PyFlyt quadrotor dynamics')
        for sc in swarm._scouts:
            sc._agent.set_physics_backend('pyflyt', mass=0.03, arm_length=0.03)
    else:
        print('[Physics] Using simple PID dynamics')

    bodies = create_drone_bodies(swarm)
    sliders = setup_sliders()
    add_arena_markers(swarm)

    hud      = HUD()
    dir_ctrl = DirectionController(
        cfg['arena_w'],
        cfg['arena_h'],
        cfg['altitude'],
    )

    logger = MovementLogger('logs')

    status = swarm.get_swarm_status()
    print(status)

    dt    = 1.0 / SIM_HZ
    frame = 0

    while True:
        # ── 0. Check connection ───────────────────────────
        if not p.isConnected():
            break

        t_start = time.perf_counter()

        try:
            # ── 1. Sliders ────────────────────────────────
            weights = read_sliders(sliders)
            swarm.set_reynolds_weights(
                weights['w_sep'], weights['w_align'],
                weights['w_coh'], weights['w_wp'],
            )

            # ── 2. Keyboard + directional control ─────────
            dir_ctrl.process(swarm, hud)

            # ── 2b. Dynamic spawn/despawn bodies ──────────
            for tier, ctrl in dir_ctrl.pending_spawns:
                pos = ctrl._agent.pos.tolist()
                r   = RADII[tier]
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                          rgbaColor=COLORS[tier])
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
                bid = p.createMultiBody(0, col, vis, pos)
                bodies.setdefault(tier, []).append(bid)
            dir_ctrl.pending_spawns.clear()
            dir_ctrl.pending_despawns.clear()

            # ── 3. Step simulation ─────────────────────────
            swarm.step(dt)

            # ── 4. Update PyBullet body positions ─────────
            update_drone_bodies(bodies, swarm)

            # ── 5. Draw hierarchy lines ───────────────────
            if frame % 3 == 0:
                draw_hierarchy_lines(swarm)

            # ── 6. HUD ────────────────────────────────────
            if frame % 6 == 0:
                status = swarm.get_swarm_status()
                hud.update_main(status, dir_ctrl.manual,
                                dir_ctrl._bias)
            if frame % (SIM_HZ * 10) == 0:
                summary = swarm._general._agent.get_zone_summary()
                print(f'[t={frame//SIM_HZ}s] Zone summary: {summary}')
            if frame % 4 == 0:
                hud.update_agent_labels(swarm)

            # ── 7. Log ────────────────────────────────────
            logger.record(swarm)

            # ── 8. Physics step ───────────────────────────
            p.stepSimulation()

        except p.error:
            # Window closed -- exit cleanly
            break
        except Exception as e:
            print(f'[Warning] frame {frame}: {e}')

        # ── 9. Frame budget sleep ─────────────────────────
        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, dt - elapsed))
        frame += 1

    # ── Cleanup ───────────────────────────────────────────
    logger.close()
    try:
        p.disconnect()
    except Exception:
        pass
    print(f'Simulation ended after {frame} frames.')


if __name__ == '__main__':
    main()
