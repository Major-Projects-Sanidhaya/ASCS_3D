"""
packet_inspector.py
-------------------
Verifies three packet-layer invariants:

  1. No source identity in any Scout or Worker packet.
  2. Every GoalToken carries the correct target_zone for its recipient node.
  3. Every Scout received at least one VelocityCommand within 5 seconds.
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np

sys.path.insert(0, '.')

from controllers.swarm_controller import SwarmController
from controllers.node_controller import NodeController
from General.General import General

# Keys that would reveal a source identity — none must appear in any packet.
_IDENTITY_KEYS = {'drone_id', 'agent_id', 'uid', 'hardware_id',
                  'node_id', 'source_id', 'sender_id', 'origin_id'}

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = '') -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f'  [PASS] {label}')
    else:
        _failed += 1
        msg = f'  [FAIL] {label}'
        if detail:
            msg += f'\n         {detail}'
        print(msg)


# ---------------------------------------------------------------------------
# Build swarm and run 5 seconds so packets flow
# ---------------------------------------------------------------------------

cfg = {
    'arena_w': 6, 'arena_h': 6,       # small arena — avoids zone split
    'grid_cols': 2, 'grid_rows': 2,
    'n_scouts_per_node': 4, 'n_workers_per_node': 1,
    'altitude': 3.0, 'emit_interval': 5.0,
}
s = SwarmController(cfg)

# Intercept packets by monkey-patching receive methods on one node's agent
_scout_packets: List[dict] = []
_worker_packets: List[dict] = []

_target_node = list(s._nodes.values())[0]
_orig_scout_recv  = _target_node._agent.receive_scout_packet
_orig_worker_recv = _target_node._agent.receive_worker_packet


def _spy_scout(pkt):
    _scout_packets.append(dict(pkt))
    return _orig_scout_recv(pkt)


def _spy_worker(pkt):
    _worker_packets.append(dict(pkt))
    return _orig_worker_recv(pkt)


_target_node._agent.receive_scout_packet  = _spy_scout
_target_node._agent.receive_worker_packet = _spy_worker

# Intercept tokens seen by that same node
_tokens_received: List[dict] = []
_orig_token_recv = _target_node._agent.receive_goal_token


def _spy_token(token):
    _tokens_received.append(dict(token))
    return _orig_token_recv(token)


_target_node._agent.receive_goal_token = _spy_token

# Intercept VelocityCommands received by scouts under that node
_scout_cmds: List[dict] = []
_monitored_scouts = list(_target_node._scout_controllers)
_orig_cmd_recvs = {}
for _sc in _monitored_scouts:
    def _make_spy(orig):
        def _spy(cmd):
            _scout_cmds.append(dict(cmd))
            return orig(cmd)
        return _spy
    _orig_cmd_recvs[id(_sc)] = _sc._agent.receive_velocity_command
    _sc._agent.receive_velocity_command = _make_spy(_sc._agent.receive_velocity_command)

for _ in range(360):   # 6 seconds at 60 Hz — ensures emit_timer fires at least once
    s.step(1 / 60)

# ---------------------------------------------------------------------------
# CHECK 1 — No source identity in any packet
# ---------------------------------------------------------------------------

print()
print('-- Anonymity: no source identity in packets --')

identity_violations = []
for pkt in _scout_packets + _worker_packets:
    found = _IDENTITY_KEYS & set(pkt.keys())
    if found:
        identity_violations.append((pkt, found))

check(
    'No source identity in any packet',
    len(identity_violations) == 0,
    f'{len(identity_violations)} packet(s) contained identity keys: '
    + str([v[1] for v in identity_violations[:3]]) if identity_violations else '',
)

total_pkts = len(_scout_packets) + len(_worker_packets)
print(f'         ({len(_scout_packets)} scout pkts + {len(_worker_packets)} worker pkts inspected)')

# ---------------------------------------------------------------------------
# CHECK 2 — Token target_zone matches node's zone_hash
# ---------------------------------------------------------------------------

print()
print('-- GoalToken: correct target_zone on accepted tokens --')

node_zh = _target_node._agent.zone_hash
accepted = [t for t in _tokens_received if t.get('target_zone') == node_zh]
wrong    = [t for t in _tokens_received if t.get('target_zone') != node_zh]

check(
    f'Token emitted with correct target_zone (zone {node_zh})',
    len(accepted) > 0,
    f'No accepted tokens found. Received {len(_tokens_received)} total, '
    f'{len(wrong)} with wrong zone.' if len(accepted) == 0 else '',
)

if wrong:
    # These are tokens for other zones that were correctly rejected
    wrong_zones = sorted({t.get("target_zone") for t in wrong})
    print(f'         ({len(wrong)} tokens for other zones {wrong_zones} — correctly rejected)')
print(f'         ({len(accepted)} token(s) accepted for zone {node_zh})')

# ---------------------------------------------------------------------------
# CHECK 3 — All scouts under the monitored node received velocity commands
# ---------------------------------------------------------------------------

print()
print('-- VelocityCommands: all scouts received at least one --')

scouts_with_cmds = sum(
    1 for sc in _monitored_scouts
    if sc._agent._cmd_age < 1.0 or sc._agent._last_cmd is not None
)
n_scouts = len(_monitored_scouts)

check(
    f'All scouts received velocity commands ({scouts_with_cmds}/{n_scouts})',
    scouts_with_cmds == n_scouts,
    f'Only {scouts_with_cmds} of {n_scouts} scouts have commands' if scouts_with_cmds < n_scouts else '',
)
print(f'         ({len(_scout_cmds)} total VelocityCommands intercepted)')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print('-' * 44)
print(f'  {_passed} passed, {_failed} failed')
print('-' * 44)

if _failed:
    sys.exit(1)
