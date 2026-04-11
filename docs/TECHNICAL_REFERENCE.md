# ASCS_3D Technical Reference

---

## 1. Architecture Overview

### Why 3 Tiers?

ASCS_3D uses a strict 3-tier hierarchy (General → Node → Scout/Worker). The prior design had 4 tiers: General → Sergeant → Corporal → Scout/Worker. This was collapsed to 3.

**Why Sergeant+Corporal collapsed into Node**

The Sergeant handled tactical decisions (zone-level mode selection, Reynolds weight adjustment). The Corporal handled execution (direct command dispatch to Scouts and Workers). Together they introduced:

1. **Double-hop latency** — every GoalToken from General had to traverse General → Sergeant → Corporal before reaching a Scout. At 60 Hz this costs 2 full ticks before any command reaches the edge.
2. **Anonymous handoff failure** — Sergeant had to know which Corporal was responsible for which Scouts, creating an identity mapping between hardware and role. This directly breaks the anonymity invariant: if Corporal 3 is responsible for Scouts A, B, C, then capturing Corporal 3 reveals the identity of those three Scouts.

Merging both into Node eliminates both problems. Node receives GoalTokens, runs Reynolds, and dispatches commands in a single tier. The zone_hash is the only identity that propagates.

**Why no peer communication at Tier 2**

Nodes do not communicate laterally with other Nodes. Horizontal coordination at Tier 2 was considered and rejected for three reasons:

1. **Attack surface reduction** — compromising one Node in a peer mesh exposes the neighbour graph, revealing zone topology. An adversary can then target the nodes responsible for high-value zones.
2. **Scaling complexity** — N nodes in a mesh require O(N²) potential links and a discovery protocol. Adding nodes requires reconfiguration. In the zone-hash model, adding nodes requires only changing two numbers in config.
3. **Anonymity preservation** — peer links would require Node addressing. Even if encrypted, the existence of a link between Node_A and Node_B is itself information.

**Sensor/Compute Inversion Table**

| Tier     | Sensors                        | Compute          | Communication role        |
|----------|-------------------------------|------------------|--------------------------|
| General  | None (no hardware)            | Full FSM + ML    | Zone-level planning       |
| Node     | Aggregate (via packets)       | Attention + RL   | Cluster coordination      |
| Scout    | Full (UWB, IMU, ToF, flow)    | Firmware PID     | Sensing + packet emission |
| Worker   | Minimal (proximity, IMU)      | FSM only         | Actuation + status report |

Compute capability decreases as you descend. Sensor data density increases as you descend. Expensive reasoning runs once per zone (Node) or once for the whole arena (General), never per-drone.

**The Anonymity Model**

Every agent interacts with the system through zone_hash integers, not hardware identifiers:

```
Physical drone hardware  →  firmware abstraction  →  zone_hash (int)
```

No uplink or downlink packet contains a source hardware address. The `zone_hash` identifies **where** in the arena an agent is operating, not **which physical unit** it is. A drone that moves between zones changes its effective identity. Packets are never sorted before aggregation (sorting would correlate position in the aggregation buffer with drone identity across ticks). HMAC with a shared swarm key authenticates that a packet came from _a_ swarm member without revealing _which_ one.

**Command/Data Flow**

```
General
  │  GoalToken (down, zone-addressed, broadcast to all Nodes)
  ▼
Node
  │  VelocityCommand (down, to Scouts)    ScoutPacket (up, anonymous)
  │  TaskCommand (down, to Workers)       WorkerPacket (up, anonymous)
  ├──────────────────────────────────────►  Scouts
  └──────────────────────────────────────►  Workers
  │
  │  ClusterStateReport (up, zone_hash keyed, no drone IDs)
  ▼
General
```

---

## 2. Class Reference — ZoneMap

**File:** `General/ZoneMap.py`

**Role:** Sole owner of all arena geometry. All other classes hold opaque `int` zone hashes and never inspect zone geometry directly.

### Constructor

`ZoneMap(arena_w: float, arena_h: float, grid_cols: int, grid_rows: int)`

- `arena_w / arena_h` — arena dimensions in metres; arena is centred at world origin `(0, 0)`
- `grid_cols / grid_rows` — initial grid partition count; `grid_cols=2, grid_rows=2` creates 4 zones

**Zone hash formula:** `h = row * grid_cols + col`

Zone 0 occupies the bottom-left cell (row=0, col=0). Hashes start at 0 and are dense for the initial grid. After split/merge operations, synthetic hashes are allocated sequentially above `grid_cols * grid_rows`.

### Public Methods

**`zone_hash(pos: np.ndarray) → int`**

Returns the active zone hash containing world-space position `pos`. Only the first two elements (x, y) are used. Returns `-1` if `pos` is outside the arena. After splits or merges the grid cell hash may be inactive; in that case, all active synthetic zones are searched by bounding-box containment. This walk is O(N zones) but occurs only when the grid topology has changed.

**`get_zone_bounds(zone_hash: int) → Tuple[ndarray, ndarray]`**

Returns `(min_corner, max_corner)` as 2-D numpy array copies. Raises `KeyError` if `zone_hash` is not in the registry (use `active_zones()` to guard).

**`get_zone_centre(zone_hash: int) → np.ndarray`**

Returns `(min_corner + max_corner) / 2` as a 2-D array `[x, y]`.

**`boundary_proximity(pos: np.ndarray, zone_hash: int) → float`**

Returns a scalar in [0, 1]. Formula: `clip(1 - min_dist_to_any_edge / BOUNDARY_RADIUS, 0, 1)`. Value is 0 at the geometric centre and 1 at or beyond any edge. Used by Node to boost separation weight near zone boundaries.

**`get_adjacent_zones(zone_hash: int) → List[int]`**

Returns hashes of active zones sharing an edge with `zone_hash`. Adjacency is defined by bounding-box geometry: two zones are adjacent when their boxes touch along one axis while overlapping (not merely touching) along the other. This definition works for both uniform grid cells and arbitrary synthetic zones produced by split/merge.

**`split_zone(zone_hash: int) → Tuple[int, int]`**

Bisects the zone along its longer axis (width vs height), creating two child zones. The parent is marked inactive. Two synthetic hashes are allocated (`_next_synthetic_hash += 2`). Returns `(child_a_hash, child_b_hash)`. After calling, `active_zones()` will include the children but not the parent.

**`merge_zones(a: int, b: int) → int`**

Merges two active, adjacent zones into one. The new bounding box is the union of both. A single synthetic hash is allocated. Both originals are marked inactive. Raises `ValueError` if either is inactive or they are not adjacent.

**`active_zones() → List[int]`**

Returns a sorted list of all active zone hashes. Sorted for determinism (zone-hash iteration order must not carry information about creation order).

**`zone_area(zone_hash: int) → float`**

Returns `(width * height)` in m².

**`needs_split(zone_hash, scout_count, coverage, collision_risk, node_health) → bool`**

Returns True if any one condition is met:
- `scout_count > SPLIT_THRESHOLD (12)` — too many scouts to coordinate in one zone
- `coverage < 0.5 AND zone_area > MIN_ZONE_AREA * 2 (50 m²)` — zone is large and under-explored
- `collision_risk > 0.7` — drones are too dense
- `node_health < 0.4` — Node is degraded; splitting may help redistribute load

**`needs_merge(a, b, density_a, density_b, time_below_min) → bool`**

Returns True only when ALL conditions are met:
- `density_a < MIN_DENSITY (2)` — zone a is under-populated
- `density_b < MIN_DENSITY (2)` — zone b is under-populated
- `time_below_min > MERGE_TIMEOUT (30.0 s)` — both have been sparse for a sustained period
- `b in get_adjacent_zones(a)` — zones share an edge

### Constants

| Name | Value | Reason |
|------|-------|--------|
| `MIN_ZONE_AREA` | 25.0 m² | A 5×5 m zone is the minimum useful patrol area for a drone cluster |
| `SPLIT_THRESHOLD` | 12 scouts | Empirically: UWB range saturates and packet aggregation overhead rises above 12 sources |
| `MERGE_TIMEOUT` | 30.0 s | Long enough to confirm the scout-count drop is permanent, not a transient communication gap |
| `MIN_DENSITY` | 2 agents | One agent per zone is wasteful and produces no inter-agent data for aggregation |
| `BOUNDARY_RADIUS` | 5.0 m | ~2× typical drone separation radius; produces a smooth proximity gradient |

---

## 3. Class Reference — General

**File:** `General/General.py`

**Role:** Top-level mission planner. Owns one ZoneMap. Maintains a zone-level world model. Issues GoalTokens. Drives zone topology changes. Never knows individual drone identities.

### Constructor

`General(position: np.ndarray, zone_map: ZoneMap, emit_interval: float = 5.0)`

Initialises the world model for every active zone with default values:
```python
{
    'centroid':       [centre_x, centre_y, 3.0],  # [x, y, z] float list
    'health':         1.0,                         # zone health 0–1
    'coverage':       0.0,                         # fraction explored 0–1
    'scout_count':    0,                           # scouts reporting from this zone
    'collision_risk': 0.0,                         # estimated collision probability
    'status':         'ACTIVE',                    # ACTIVE | DEGRADED | SILENT
    'last_seen':      0.0,                         # uptime of last received report
    'silence_timer':  0.0,                         # seconds since last report
}
```

Builds `_waypoint_sequence`: 5 waypoints at the four arena corners (at 80% of half-width) and centre, all at altitude 3.0 m.

### FSM Phases and Transitions

```
EXPLORE  (15 s) → CONVERGE (10 s) → EXECUTE (20 s) → WITHDRAW (8 s) → REBALANCE (12 s) → EXPLORE
```

Transitions are time-based. `handle_silent_zone` can force a jump to REBALANCE if >50% of zones are silent.

Phase-to-mode mapping:

| Phase | Node Mode |
|-------|-----------|
| EXPLORE | SCOUT |
| CONVERGE | CONVERGE |
| EXECUTE | HOLD |
| WITHDRAW | WITHDRAW |
| REBALANCE | SCOUT |

### Key Methods

**`update_zone(zone_hash, report)`**

Ingests a ClusterStateReport. Updates `centroid`, `health`, `coverage` (from `coverage_fraction`), `scout_count`, `collision_risk`, `last_seen`. Resets `silence_timer` to 0.0 and `status` to ACTIVE. Creates a default entry if `zone_hash` is not already in the model.

**`tick_silence_timers(dt)`**

Called every tick. Increments `silence_timer` for every zone by `dt`. Applies thresholds:
- `silence_timer >= SILENCE_DEAD (10.0 s)` → status = SILENT, calls `handle_silent_zone`
- `silence_timer >= SILENCE_WARN (2.0 s)` → status = DEGRADED
- otherwise → status = ACTIVE

Note: `update_zone` resets `silence_timer` to 0, so zones that report each tick never accumulate.

**`build_goal_token(zone_hash, waypoint, mode) → dict`**

```python
{
    'target_zone': zone_hash,           # int — the zone this token is addressed to
    'waypoint':    waypoint.tolist(),   # [x, y, z] — zone-level objective
    'mode':        mode,                # str — SCOUT|HOLD|CONVERGE|WITHDRAW
    'priority':    1 - min(health, coverage),  # float — urgency 0–1
    'ttl':         emit_interval * 1.2, # float s — token expires if not refreshed
    'threat_mask': self._threat_mask,  # int — bitmask of hostile zones
    'timestamp':   self._uptime,       # float — General's simulation uptime
}
```

`ttl = emit_interval * 1.2` gives a 20% grace window above the nominal emit interval. This prevents Node from entering autonomous hold due to a single missed broadcast cycle.

**`broadcast_tokens(nodes)`**

Iterates `active_zones()`, builds one token per zone, and calls `node.receive_goal_token(token)` on **every** node in the list. Nodes self-select by checking `token['target_zone'] == self.zone_hash` and discarding non-matching tokens. This broadcast-then-filter design means General never needs to know which Node covers which zone — that mapping lives only inside each Node.

**`check_zone_splits()`** and **`check_zone_merges()`**

Called every tick. Inspect `_world_model` data against `ZoneMap` predicates. On split: creates world model entries for both children, removes parent entry. On merge: creates world model entry for the merged zone (averaged health/coverage), removes both originals.

---

## 4. Class Reference — Node

**File:** `Node/Node.py`

**Role:** Mid-tier coordinator. Receives GoalTokens, aggregates Scout/Worker packets via attention, runs Reynolds flocking, dispatches commands, reports cluster state upward.

### Constructor

`Node(position: np.ndarray, zone_map_ref: ZoneMap, zone_hash: int, altitude: float = 3.0)`

### Reynolds Translation Table

| Mode     | w_sep | w_align | w_coh | Behavioural intent |
|----------|-------|---------|-------|-------------------|
| SCOUT    | 0.8   | 0.4     | 0.2   | Spread out, low cohesion — cover maximum area |
| HOLD     | 0.5   | 0.8     | 0.6   | Stable formation, high alignment — hover in place |
| CONVERGE | 0.3   | 0.7     | 0.9   | Pull tightly together — focus on a target |
| DISPERSE | 1.0   | 0.3     | 0.1   | Maximum separation — emergency spread |
| WITHDRAW | 0.6   | 0.9     | 0.7   | Move together — coordinated retreat |

### Attention Aggregator (`aggregate_observations`)

Pipeline (numpy only, no PyTorch):

1. **Feature extraction** — 10 floats per packet: `[rel_pos_mean(3), rel_hdg_mean(3), speed(1), obs_fwd(1), obs_min(1), battery(1)]`
2. **Embed** — `E = X @ W_embed` where `W_embed` is a fixed random (10, 32) matrix (seed=42, scale 0.1). Shape: `(N, 32)`.
3. **Scaled dot-product attention** — `logits = E @ E.T / sqrt(32)`, subtract row max for numerical stability, apply softmax row-wise to get attention matrix `A`. Shape: `(N, N)`.
4. **Attend** — `attended = A @ E`. Shape: `(N, 32)`.
5. **Mean pool** — `pooled = attended.mean(axis=0)`. Shape: `(32,)`. Mean pooling (not max) is used because it is permutation-invariant and produces stable gradients during training.
6. **Project** — `result = pooled @ W_proj` where `W_proj` is a fixed random (32, 64) matrix (seed=43, scale 0.1). Shape: `(64,)`.

Seeds 42 and 43 are fixed at module level so the same projection is used across all Node instances and simulation runs. In production, these matrices would be replaced with trained PyTorch parameters exported to ONNX.

**Why packets must never be sorted before aggregation:**

TDMA slot assignment is re-randomised every 10 seconds. Over many ticks, sorting by any field (timestamp, seq, speed) would create a stable ordering. A Node that observes many ticks of sorted data could reconstruct which sequence position corresponds to which physical drone. Mean pooling over unsorted packets is permutation-invariant by construction — no ordering information survives the aggregation.

### `get_effective_weights`

Computes the final (w_sep, w_align, w_coh) triple:

1. Base weights from `REYNOLDS_TABLE[self._mode]` (falls back to HOLD if token expired)
2. Add RL offsets: `w_sep += delta_sep`, etc. Each clamped to [0, 1].
3. Boundary boost: `w_sep = clip(w_sep + boundary_proximity(pos, zone_hash) * 0.4, 0, 1.5)`

The boundary boost is additive (not multiplicative) so it activates smoothly as the Node approaches a zone edge and does not overwhelm the base weights in the zone interior.

### `compute_velocity`

```python
v = (w_sep   * reynolds_separation(neighbor_pos)
   + w_align * reynolds_alignment(neighbor_vel)
   + w_coh   * reynolds_cohesion(neighbor_pos)
   + w_wp    * reynolds_waypoint_pull(waypoint))
```

Horizontal speed clamped to MAX_SPEED (4.0 m/s). Z channel overridden: `v[2] = (ALTITUDE - pos[2]) * 2.0` — a proportional altitude hold independent of the flocking forces.

`w_wp = 0.35` default. This gives waypoint pull enough influence to navigate the swarm across the zone without overriding the inter-agent forces that prevent collision.

### `set_rl_offsets(delta_sep, delta_align, delta_coh)`

Offsets are clamped to [-0.25, 0.25]. This bound serves two purposes:

1. **Safety** — the translation table provides a reasonable baseline; the RL policy cannot deviate far enough from it to cause dangerous behaviour during early training.
2. **Training stability** — unbounded policy output in weight space creates a non-stationary target; the bound normalises the action space.

### `handle_obstacle_reflex`

```python
if obs_min < 0.5:
    return -vel / (|vel| + 1e-6) * MAX_SPEED
```

This runs **before** the policy's velocity is applied. The reflex overrides everything. It is not part of the RL policy because: (a) it must be instantaneous (no inference latency), and (b) the training penalty for collision is already sufficient — the policy learns to avoid triggering the reflex.

### Degradation Cascade

- `_token_age` increments every tick via `update_position(dt)`.
- `_token_age >= TTL_WARN (2.0 s)` — token is stale; logged by NodeController silence timer.
- `_token_age >= TTL_DEAD (10.0 s)` — `handle_general_silence()` called; mode = HOLD, waypoint frozen at current position. Node operates autonomously.

---

## 5. Class Reference — Scout

**File:** `Scout/Scout.py`

**Role:** Sensing tier. Reads all sensors, emits anonymous ScoutPackets, executes VelocityCommands via PID.

### Hardware Sensor Mapping

| Simulation method | Real hardware | Notes |
|---|---|---|
| `read_uwb_ranges(all_scouts)` | Qorvo DW3000 UWB radio | 50 m range, ~3 cm noise σ |
| `read_imu()` | TDK ICM-42688-P 6-axis IMU | Returns heading unit vector + velocity |
| `read_barometer()` | Bosch BMP388 barometric altimeter | Returns z-altitude in metres |
| `read_tof_obstacle()` | STMicroelectronics VL53L5CX 8×8 ToF | Returns (fwd_dist, min_dist); placeholder 5.0 m in sim |
| `read_optical_flow()` | PMW3901 optical flow sensor | Returns 2-D lateral velocity estimate |
| `read_env_sensors()` | 4-channel environmental sensor array | CO₂, temperature, humidity, light; zeros in sim |
| `motor_mix(v_cmd)` | 4× brushless ESC + flight controller | Blends vx/vy/vz/yaw-rate into per-motor throttle |

### ScoutPacket Schema

```python
{
    'swarm_id':      1,         # int — identifies swarm, NOT drone
    'seq':           int,       # 0–65535 wrapping; deduplication only
    'rel_positions': list,      # list of [x, y, z] relative vectors to nearby scouts
    'rel_headings':  list,      # list of [hx, hy, hz] unit vectors
    'speed':         float,     # m/s
    'obs_fwd':       float,     # m — forward obstacle distance
    'obs_min':       float,     # m — minimum obstacle distance
    'env':           list,      # 4-element float environment vector
    'battery':       float,     # 0–1 fraction
    'timestamp':     float,     # Scout's _uptime (NOT wall-clock)
    # NO 'source_id' field — anonymity guarantee
}
```

### `pid_update(v_target, dt)`

```python
err       = v_target - self.vel
self.vel += err * min(dt * 8.0, 1.0)
```

The gain `8.0/s` gives a time constant of approximately 0.125 s. `min(dt * 8.0, 1.0)` prevents overshoot for large dt values (e.g., if the simulation runs a catch-up tick after lag). At 60 Hz, `dt * 8.0 = 0.133`, well within the stable range.

### LOITER_TIMEOUT = 0.5 s

When `_cmd_age >= 0.5 s` (no fresh command received), Scout enters loiter:
- Lateral velocity damped: `vel[:2] *= 0.9` per tick (exponential decay to zero)
- Altitude hold: `vel[2] = (pos[2] - pos[2]) * 2.0 = 0` (hold current altitude)

This prevents Scouts from drifting unpredictably when Node communication is interrupted. The 0.5 s threshold is set below the Node's VelocityCommand TTL of 0.5 s, so a single missed packet triggers loiter — conservative by design.

### TDMA Simulation Note

The `seq` field wraps at 65536 and is used by the Node only for deduplication (dropping duplicate packets in a single tick). It is never used to identify a source. In real hardware, TDMA slot assignment re-randomises every 10 s, implemented as jitter in the `timestamp` field in simulation.

---

## 6. Class Reference — Worker

**File:** `Worker/Worker.py`

**Role:** Actuation tier. Executes physical tasks on command. Minimal sensors. Emits anonymous WorkerPackets.

### Five-State Task FSM

```
IDLE ──(receive TaskCommand)─���► MOVING
MOVING ──(dist < 0.3 m)──────► EXECUTING
EXECUTING ──(success)────────► COMPLETE
EXECUTING ──(timeout 10 s)───► FAILED
COMPLETE / FAILED ──(next tick)► IDLE
Any non-IDLE ──(cmd_age > CMD_TIMEOUT)► IDLE
```

`CMD_TIMEOUT = 1.0 s` — a Worker that has not received a refreshed command for 1 second reverts to IDLE. This is much tighter than Node's TTL_DEAD (10 s) because Workers are executing physical actions (moving payloads) where stale commands are immediately dangerous.

`ARRIVAL_THRESHOLD = 0.3 m` — arrival is declared when the Worker is within 0.3 m of `target_pos`. This is approximately one drone body length, providing a stable arrival detection without oscillation.

### `execute_payload_action`

| Payload type | Logic | Completion condition |
|---|---|---|
| `generic` | No-op | Always True immediately |
| `gripper` | `payload_state = 1 if grip_state else 0` | Always True immediately |
| `dispenser` | `payload_state = 0` on release | True only if `params['release']` is True; waits otherwise |
| `relay` | `payload_state = 1 if active else 0` | Always True immediately |

For `dispenser`, returning False while waiting for the `release` command means the FSM remains in EXECUTING. This is intentional: the Worker holds position and waits for an updated TaskCommand containing `release: True`.

### Comparison to Scout

| Dimension | Scout | Worker |
|---|---|---|
| Primary function | Sensing | Actuation |
| Command type | VelocityCommand | TaskCommand |
| State machine | Stateless (PID loop) | 5-state FSM |
| Payload | None | gripper/dispenser/relay/generic |
| Packet uplink | ScoutPacket | WorkerPacket |
| Position control | PID toward v_target | PID toward target_pos |

---

## 7. Controller Architecture

### BaseController (`controllers/base_controller.py`)

Abstract base for all four tier controllers. Provides fully implemented shared utilities.

**8 abstract methods (must be implemented by each tier):**

| Method | Purpose |
|--------|---------|
| `step(dt)` | Advance controller by one simulation tick |
| `receive_downlink(packet)` | Accept incoming packet from the tier above |
| `send_uplink(packet)` | Dispatch packet to the tier above |
| `receive_uplink(packet)` | Accept incoming packet from the tier below |
| `send_downlink(packet)` | Dispatch packet to the tier below |
| `get_status() → dict` | Return tier-specific status snapshot |
| `handle_silence(source, dt)` | React to communication gap from named source |
| `reset()` | Return controller to initial state |

**Shared concrete utilities:**

- `log_event(type, payload)` — structured event log, evicts oldest entry beyond MAX_LOG_SIZE=500
- `tick_uptime(dt)` — advances `_uptime` and `_step_count`; call once at the start of each tick
- `update_silence_timer(source, dt, received) → str` — returns STATUS_HEALTHY/DEGRADED/SILENT; resets timer if `received=True`, increments otherwise
- `is_packet_expired(packet, now) → bool` — compares `now - timestamp > ttl`; returns False if fields absent
- `validate_packet(packet, required_keys) → Tuple[bool, str]` — type check + key presence
- `build_base_status() → dict` — returns common fields: tier, agent_id, uptime, step_count, status, silence_timers, log_length
- `clamp(v, lo, hi)`, `normalize(vec)` — math helpers

### GeneralController

Wraps `General`. `step(dt)` order: tick_uptime → update_position → tick_silence_timers → run_fsm_step → check_zone_splits → check_zone_merges → broadcast_tokens. `register_node(nc)` must be called before the first `step()` for each NodeController.

### NodeController

Wraps `Node`. `step(dt)` order: tick_uptime → clear_packet_buffer → scouts emit → workers emit ��� aggregate_observations → compute_velocity (or obstacle reflex) → broadcast_scout_commands → broadcast_worker_commands → update_position.

`register_scout(sc)` appends to both `_scout_controllers` (controller list) and `_agent._scouts` (agent list). Both lists must stay in sync.

### ScoutController / WorkerController

Thin wrappers. `step(dt)`: tick_uptime → update_position → battery check → silence detection.

### SwarmController (`controllers/swarm_controller.py`)

Top-level orchestrator. `step(dt)` order:

1. `self._general.step(dt)` — General ticks, tokens broadcast
2. For each Node: `set_gui_weights`, `node.step(dt)`, **`node._agent.send_report(self._general._agent)`**
3. For each Scout: `scout.step(dt)`
4. For each Worker: `worker.step(dt)`

**Why `node.send_report()` is called from `SwarmController.step()`, not `NodeController.step()`:**

`send_report` writes into General's `_world_model`. If placed in `NodeController.step()`, NodeController would need a reference to GeneralController at construction time. This creates a circular dependency at construction: GeneralController holds NodeController references (for token broadcast), and NodeController would need to hold a GeneralController reference (for report delivery). The dependency graph becomes a cycle.

SwarmController already owns both, so it bridges the call after each Node tick completes. This also enforces correct temporal ordering: General always reads state from the fully-completed previous Node tick, never from a partially-updated Node.

---

## 8. Communication Protocols

### GoalToken (General → Node, broadcast)

```python
{
    'target_zone': int,    # zone_hash — the zone this token addresses
    'waypoint':    list,   # [x, y, z] float — zone-level objective position
    'mode':        str,    # SCOUT | HOLD | CONVERGE | WITHDRAW
    'priority':    float,  # 0–1 — urgency; 1 - min(health, coverage)
    'ttl':         float,  # seconds — emit_interval * 1.2
    'threat_mask': int,    # bitmask — hostile zone hashes (0 = no threats)
    'timestamp':   float,  # General's simulation uptime
}
```

Deliberately absent: node hardware address (broadcast to all; self-selected), drone count (General never knows), formation spec (Node decides how to spread subordinates).

### ClusterStateReport (Node → General)

```python
{
    'zone_hash':         int,   # int — identifies the reporting zone
    'centroid':          list,  # [x, y, z] — estimated cluster centroid
    'health':            float, # 0–1 — Node health estimate
    'coverage_fraction': float, # 0–1 — fraction of zone with scout coverage
    'scout_count':       int,   # number of Scouts in this cluster
    'collision_risk':    float, # 0–1 — estimated collision probability
    'velocity_mean':     list,  # [vx, vy, vz] mean cluster velocity
    'timestamp':         float, # Node's simulation uptime
}
```

Deliberately absent: hardware IDs of any Scout or Worker.

### VelocityCommand (Node → Scout)

```python
{
    'v_target':  list,   # [vx, vy, vz] m/s — target velocity
    'speed_max': float,  # m/s — cap; currently MAX_SPEED = 3.0
    'ttl':       float,  # 0.5 s — Scout enters loiter if exceeded
    'timestamp': float,  # Node's uptime
}
```

### TaskCommand (Node → Worker)

```python
{
    'action':     str,   # MOVE_TO | EXECUTE_TASK | RETURN | HOVER
    'target_pos': list,  # [x, y, z] — world-space target position
    'params':     dict,  # payload-specific: {grip_state, release, active, ...}
    'ttl':        float, # 5.0 s
    'timestamp':  float,
}
```

### ScoutPacket (Scout → Node)

Documented in Section 5. Key absence: `source_id`.

### WorkerPacket (Worker → Node)

```python
{
    'swarm_id':      int,    # 1 — swarm identifier
    'seq':           int,    # 0–65535 wrapping; deduplication only
    'task_status':   str,    # IDLE | MOVING | EXECUTING | COMPLETE | FAILED
    'pos_rel':       list,   # [0, 0, 0] placeholder (no GPS)
    'proximity_min': float,  # 5.0 m placeholder
    'battery':       float,  # 0–1
    'payload_state': int,    # 0 or 1 binary actuator state
    'timestamp':     float,
    # NO source_id
}
```

### HMAC Authentication

`Scout.sign_packet(packet, swarm_key)` and `Worker.sign_packet(packet, swarm_key)` append a 2-byte BLAKE2b digest:

```python
data = str(sorted(packet.items())).encode()
h    = hashlib.blake2b(data, key=swarm_key[:32], digest_size=2).digest()
```

The shared `swarm_key` is pre-deployed on all hardware. The HMAC proves packet origin is *a swarm member* without revealing *which* one. digest_size=2 (16 bits) is minimal — sufficient for spoofing detection in a small controlled swarm without identifying senders. In a higher-security deployment, digest_size should be 16 or 32.

---

## 9. Reynolds Flocking

### The Four Forces

| Force | Formula | Effect |
|-------|---------|--------|
| Separation | `Σ(diff / (|diff|² + ε))` for `|diff| < SEP_RADIUS` | Prevent collision |
| Alignment | `mean(neighbour_vel) - self.vel` | Match flock direction |
| Cohesion | `mean(neighbour_pos) - self.pos` | Stay in group |
| Waypoint pull | `(wp - pos) / (|wp - pos| + ε)` | Move toward zone objective |

Combined: `v = w_sep·F_sep + w_align·F_align + w_coh·F_coh + w_wp·F_wp`

Horizontal speed clamped to MAX_SPEED. Z channel set independently via altitude hold.

### Why RL Outputs Offsets, Not Raw Velocities

The translation table provides a safe, interpretable baseline that guarantees reasonable behaviour before training. If the RL policy output raw velocities, an untrained policy would produce garbage velocities immediately. By outputting offsets bounded to [-0.25, 0.25], the policy can only modify the weight table by a maximum of 25% — ensuring the translation table dominates and the swarm never enters a completely unsafe state due to an untrained policy.

This is equivalent to a residual policy architecture: `effective_weights = table_weights + policy_output`.

### Permutation Invariance Requirement

Scout packets arrive in TDMA slots that are re-randomised every 10 s. Any sort operation on packets before aggregation would correlate packet position (first, second, third...) with sender identity across ticks. Over hundreds of ticks, even a weak correlation is sufficient to reconstruct drone identities. Mean pooling is permutation-invariant by construction — the output is the same regardless of input order. Scaled dot-product attention is also permutation-invariant when the output is pooled (not read positionally).

**Never sort `_last_packets`.** This invariant is enforced by code convention — no sort call exists anywhere in the aggregation pipeline.

### Boundary Proximity Boost

```python
prox  = zone_map.boundary_proximity(pos, zone_hash)   # 0 at centre, 1 at edge
w_sep = clip(w_sep + prox * 0.4, 0, 1.5)
```

`BOUNDARY_SEP_BOOST = 0.4`. At the zone boundary, separation weight increases by up to 0.4 additively. This prevents drones from wandering into adjacent zones (which would break the zone-based reporting invariant) without needing a hard wall. The effect is a smooth inward pressure that increases as the drone approaches the edge.

---

## 10. ZoneMap and Scaling

### Zone Hash Model

```
arena_w=20, arena_h=20, grid_cols=2, grid_rows=2

Zone 0 (row=0, col=0): x∈[-10, 0],  y∈[-10, 0]   ← bottom-left
Zone 1 (row=0, col=1): x∈[0,  10],  y∈[-10, 0]   ← bottom-right
Zone 2 (row=1, col=0): x∈[-10, 0],  y∈[0,  10]   ← top-left
Zone 3 (row=1, col=1): x∈[0,  10],  y∈[0,  10]   ← top-right
```

**How General addresses Nodes without knowing their identity:**

General builds one GoalToken per `zone_hash` and calls `node.receive_goal_token(token)` on every registered node. Each Node's `receive_goal_token` returns `False` immediately if `token['target_zone'] != self.zone_hash`. No central dispatch table exists. The routing table is implicit in each Node's zone assignment.

### Split and Merge

**Split procedure:**
1. Find longer axis (width vs height) of target zone
2. Bisect at midpoint; create two child zones with synthetic hashes
3. Mark parent inactive
4. General creates world model entries for children; removes parent entry
5. In the next tick, `active_zones()` returns the children; General broadcasts tokens to them

**Merge procedure:**
1. Verify both zones active and adjacent
2. New bounding box = union of both boxes
3. Register new zone with synthetic hash
4. Mark both originals inactive
5. General creates merged world model entry (averaged health/coverage); removes both originals

### UWB Range Constraint

UWB ranging in simulation is limited to 50 m (`UWB_MAX_RANGE`). For a zone of size L × L, scouts at zone corners are `L√2` apart. For all scouts to maintain UWB contact: `L√2 < 50 m`, meaning `L < 35 m`. Zones larger than ~35 m per side will produce scouts that cannot range each other across the zone diagonal.

### Arena Size to Node Count Table

| Grid | Nodes | Scout/Worker agents (4 scouts + 1 worker per node) | Total agents |
|------|-------|---------------------------------------------------|-------------|
| 2×2  | 4     | 16 + 4 = 20                                       | 25          |
| 3×3  | 9     | 36 + 9 = 45                                       | 55          |
| 4×4  | 16    | 64 + 16 = 80                                      | 97          |
| 5×5  | 25    | 100 + 25 = 125                                    | 151         |

---

## 11. Anonymity Guarantees

### Every Potential Identity Leak and Its Mitigation

| Potential leak | Mitigation |
|---|---|
| Hardware MAC in radio packet | Not transmitted; HMAC key is shared, not per-device |
| GPS absolute position | No GPS hardware on Scout or Worker |
| Consistent packet ordering | TDMA re-randomisation + mean pooling in aggregator |
| Source address in packet header | No source field in ScoutPacket or WorkerPacket |
| Timing correlation via consistent timestamps | Scout timestamp = simulation uptime, not wall-clock |
| Zone assignment tied to specific drone | Nodes are assigned to zones, not to drones |
| Node-to-Node topology exposure | No Node-to-Node communication |
| `seq` field used for identification | `seq` used for deduplication only, not identity |

### Hardware vs Software Anonymity

**Software anonymity** (this system): no identifier in any transmitted packet; attention aggregator is permutation-invariant; zone_hash is geometric, not hardware-linked.

**Hardware anonymity** (deployment requirement beyond this codebase): GPS module must be physically removed or RF-shielded; radio transmitter must not broadcast hardware serial number in its PHY preamble; firmware must disable JTAG enumeration in flight mode; boot ROM must not expose device UID via any radio interface.

### Why zone_hash Is Not an Identity

A `zone_hash` identifies a geometric region, not a drone. After a split, both children get new hashes — the physical drone that was operating in the parent zone now operates under a new zone_hash. After a merge, the merged zone hash is different from either original. The same physical drone may be addressable under three different zone hashes over the course of a mission (original, split child, merged result), providing forward anonymity: capturing a drone at time T reveals its zone_hash at time T, but not its zone_hash at time T-1 or T+1 if topology changes occurred.

---

## 12. Training Curriculum

### Phase 1 — Encoder Warm-Start (Behavioural Cloning)

- **What trains:** `W_embed` (10×32), `W_proj` (32×64) in the attention aggregator
- **What freezes:** Reynolds weights (fixed to translation table values)
- **Training signal:** Cross-entropy loss against mode labels derived from the translation table
- **Duration:** ~100k steps at 60 Hz simulation
- **Purpose:** Teach the attention encoder to produce cluster embeddings that capture behavioural state (crowded vs sparse, aligned vs spread) before RL destabilises the weights

### Phase 2 — Separation Safety (RL, frozen encoder)

- **What trains:** Policy heads for (Δw_sep, Δw_align, Δw_coh)
- **What freezes:** W_embed, W_proj
- **Reward:** `+1` per step without collision, `-10` on collision, `+0.1` for centroid within zone bounds
- **Algorithm:** PPO (proximal policy optimisation); clip ratio 0.2
- **Purpose:** Learn safe flocking before optimising task performance

### Phase 3 — Coverage and Waypoint Tracking (all trainable)

- **Additional reward terms:** `+0.5 * coverage_fraction`, `+0.3 * (1 - dist_to_waypoint / zone_diameter)`
- **Purpose:** Learn to efficiently cover zones and respond to GoalToken waypoints

### Phase 4 — Fault Tolerance

- **Training perturbations:** random Scout dropouts (0–20% per tick), delayed packets (up to 100 ms jitter), simulated General communication loss (up to 15 s blackouts)
- **Additional reward:** `+0.2` for maintaining cluster health > 0.7 under perturbation
- **Purpose:** Robustness to real-world unreliable communication

---

## 13. Running the Simulation

### Install

```bash
pip install numpy pybullet
```

### Run

```bash
cd ASCS_3D
python swarm_sim.py
```

### Configuration (`SWARM_CONFIG` in `swarm_sim.py`)

```python
SWARM_CONFIG = {
    'arena_w':            20.0,   # arena width in metres
    'arena_h':            20.0,   # arena height in metres
    'grid_cols':          2,      # ← change to scale horizontally
    'grid_rows':          2,      # ← change to scale vertically
    'n_scouts_per_node':  4,      # scouts per zone
    'n_workers_per_node': 1,      # workers per zone
    'altitude':           3.0,    # hover altitude in metres
    'emit_interval':      5.0,    # seconds between GoalToken broadcasts
}
```

To scale to a 3×3 grid (9 zones, 55 agents): change `grid_cols=3, grid_rows=3`. No other changes needed.

### Live Sliders

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| `w_sep` | 0–1.5 | 0.6 | Move right: agents spread apart; collision avoidance increases |
| `w_align` | 0–1.5 | 0.4 | Move right: agents match direction; flock moves as a unit |
| `w_coh` | 0–1.5 | 0.5 | Move right: agents cluster tightly around centroid |
| `w_wp` | 0–1.0 | 0.35 | Move right: stronger pull toward zone waypoint |

### Visual Output

| Object | Colour | Description |
|--------|--------|-------------|
| Large sphere | Purple `(0.6, 0.1, 0.8)` | General (hovers stationary above arena centre) |
| Medium spheres | Teal `(0.0, 0.7, 0.7)` | Nodes (one per zone) |
| Small spheres | Amber `(1.0, 0.7, 0.0)` | Scouts (ring around their Node) |
| Small spheres | Blue `(0.1, 0.4, 0.9)` | Workers (ring around their Node) |
| Short lines | Teal | Node → Scout command links (4 frame lifetime) |
| Short lines | Blue | Node → Worker command links |
| Flat cylinders on ground | Yellow | Waypoint markers |

### Expected Terminal Output

On startup:
```
[swarm_sim] Initialised — {'step_count': 0, 'total_agents': 25, 'agents_per_tier': {'GENERAL': 1, 'NODE': 4, 'SCOUT': 16, 'WORKER': 4}, 'active_zones': 4, ...}
```

Every 5 simulated seconds:
```
[t=5.0s] phase=EXPLORE  zones=4  health=1.00
[t=10.0s] phase=EXPLORE  zones=4  health=1.00
```

---

## 14. Known Limitations and Next Steps

### What Is Simulated vs Real Hardware

| Feature | Simulation | Real hardware |
|---------|-----------|--------------|
| UWB ranging | Gaussian noise on Euclidean diff | DW3000 with multipath fading |
| Obstacle detection | Placeholder 5.0 m | ToF raycasting in PyBullet / VL53L5CX |
| Battery drain | Static 1.0 | Real discharge model (C-rate dependent) |
| TDMA | Timestamp jitter | Sub-GHz radio time slots, re-randomised every 10 s |
| Motor mixing | Simplified 4-rotor arithmetic | Real ESC + autopilot (PX4 / ArduPilot) |
| Attention weights | Fixed random numpy matrices | Trained PyTorch parameters (ONNX for deployment) |
| RL offsets | 0.0 (policy not trained) | PPO-trained residual policy |

### Attention Aggregator Placeholder

`Node.aggregate_observations()` uses fixed random matrices (seeds 42, 43) as placeholders. To replace with trained weights:

1. Train `W_embed` (10×32) and `W_proj` (32×64) using the Phase 1 curriculum
2. Replace `_get_projection_matrices()` in `Node.py` with a function that loads from a checkpoint
3. Export to ONNX for deployment on Cortex-M7 or Jetson Orin NX
4. The public interface (`aggregate_observations() → ndarray[64]`) is unchanged

### Minimal Phase 1 Training Loop Skeleton

```python
import torch
import numpy as np
from Node.Node import _get_projection_matrices

W_embed_np, W_proj_np = _get_projection_matrices()
W_embed = torch.tensor(W_embed_np, requires_grad=True, dtype=torch.float32)
W_proj  = torch.tensor(W_proj_np,  requires_grad=True, dtype=torch.float32)
optimizer = torch.optim.Adam([W_embed, W_proj], lr=1e-3)

for batch in dataloader:           # batches of (N_packets, 10) feature tensors
    X      = batch['features']     # (B, N, 10)
    labels = batch['mode_label']   # (B,) int — mode index from translation table
    E      = X @ W_embed           # (B, N, 32)
    scale  = 32 ** 0.5
    logits = (E @ E.transpose(-1, -2)) / scale
    A      = torch.softmax(logits, dim=-1)
    pooled = (A @ E).mean(dim=-2)  # (B, 32)
    out    = pooled @ W_proj        # (B, 64)
    # ... head → logits over 5 modes → cross-entropy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Scaling Limits

- **UWB range** caps effective zone size at ~35 m per side (√2 diagonal constraint)
- **Attention aggregator** is O(N²) in packet count; for N > 64 packets per tick, switch to linear attention (Performer or Nyströmformer)
- **PyBullet GUI** slows above ~100 agents; use `p.connect(p.DIRECT)` for headless runs with large grids
- **split/merge** does not yet spawn/despawn NodeControllers at runtime (only ZoneMap topology changes); `SwarmController.despawn_node` is implemented but dynamic re-spawning requires additional wiring
