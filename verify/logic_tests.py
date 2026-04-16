"""
verify/logic_tests.py
---------------------
Lightweight logic tests for ASCS_3D. No external test runner required —
run directly with:

    python verify/logic_tests.py

Exit code 0 = all tests passed.  Any assertion error = test failed.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

# ── Add project root to path ──────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from controllers.swarm_controller import SwarmController
from General.ZoneMap import ZoneMap
from Node.Node import Node

# ── Minimal test harness ──────────────────────────────────────────────────────

_passed = 0
_failed = 0

HEAD = '\n{}'


def test(name: str, fn) -> None:
    global _passed, _failed
    try:
        fn()
        print(f'  [PASS] {name}')
        _passed += 1
    except Exception as exc:
        print(f'  [FAIL] {name}')
        traceback.print_exc()
        _failed += 1


# ── Worker sequencing tests ───────────────────────────────────────────────────

print(HEAD.format('-- Worker sequencing -- no premature movement --'))


def test_workers_hold_on_frame_1():
    """Workers must not move on the very first frame."""
    from Node.Node import OP_SCOUTING
    # 6x6 arena = 36 m² per zone — below MIN_ZONE_AREA*2=50 so no auto-split
    cfg = {
        'arena_w': 6, 'arena_h': 6,
        'grid_cols': 1, 'grid_rows': 1,
        'n_scouts_per_node': 4, 'n_workers_per_node': 2,
        'altitude': 3.0, 'emit_interval': 0.1,
    }
    s = SwarmController(cfg)
    worker_pos_before = [w._agent.pos.copy() for w in s._workers]
    # Run only 3 frames — not enough for coverage threshold
    for _ in range(3):
        s.step(1 / 60)
    worker_pos_after = [w._agent.pos.copy() for w in s._workers]
    for i, (before, after) in enumerate(zip(worker_pos_before, worker_pos_after)):
        dist = np.linalg.norm(after - before)
        assert dist < 0.5, (
            f'Worker {i} moved {dist:.3f} m on first 3 frames — should hold'
        )


test('Workers hold position during SCOUTING phase', test_workers_hold_on_frame_1)


def test_workers_only_move_after_coverage():
    """Workers must not receive MOVE_TO until coverage threshold is reached."""
    from Node.Node import OP_SCOUTING, OP_TASKING, COVERAGE_THRESHOLD
    # 6x6 arena = 36 m² — below MIN_ZONE_AREA*2=50 so no auto-split
    cfg = {
        'arena_w': 6, 'arena_h': 6,
        'grid_cols': 1, 'grid_rows': 1,
        'n_scouts_per_node': 6, 'n_workers_per_node': 1,
        'altitude': 3.0, 'emit_interval': 0.1,
    }
    s = SwarmController(cfg)
    node = list(s._nodes.values())[0]._agent

    reached_tasking = False
    for f in range(600):
        s.step(1 / 60)
        if node.get_op_phase() == OP_TASKING:
            reached_tasking = True
            break
        # Before TASKING, workers should only be in IDLE or MOVING
        if f > 10:
            for w in s._workers:
                assert w._agent._task_state in ('IDLE', 'MOVING'), (
                    f'Worker in unexpected state: {w._agent._task_state}'
                )

    assert reached_tasking, (
        f'Never reached TASKING after 600 frames. '
        f'Coverage: {node._coverage_fraction:.2f}, '
        f'Packets: {len(node._last_packets)}'
    )
    print(f'       (reached TASKING at frame {f}, '
          f'coverage={node._coverage_fraction:.2f})')


test('Node transitions to TASKING after coverage threshold',
     test_workers_only_move_after_coverage)


def test_op_phase_starts_scouting():
    from Node.Node import OP_SCOUTING
    zm   = ZoneMap(10, 10, 1, 1)
    node = Node(np.array([0.0, 0.0, 3.0]), zm, 0)
    assert node.get_op_phase() == OP_SCOUTING, (
        f'Node should start in SCOUTING, got {node.get_op_phase()}'
    )


test('Node starts in SCOUTING phase', test_op_phase_starts_scouting)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f'\n{"-" * 44}')
print(f'  {_passed} passed, {_failed} failed')
print(f'{"-" * 44}')

if _failed:
    sys.exit(1)
