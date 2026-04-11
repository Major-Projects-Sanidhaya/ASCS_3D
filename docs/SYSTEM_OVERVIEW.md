# ASCS_3D — System Overview

**Audience:** Someone reading this project for the first time, no assumed background in robotics or multi-agent systems.

---

## 1. What Is This System?

ASCS_3D is a simulation of a drone swarm that coordinates itself the way an ant colony does — no single drone is "in charge" of everything, no drone knows the full picture, yet together they cover a space, detect changes, and carry out tasks.

A handful of higher-level drones watch the big picture and issue general directions. A layer of mid-tier drones translate those directions into local choreography. At the bottom, small sensing drones scout the environment and actuation drones carry out physical tasks like picking up objects or dropping markers.

The whole thing runs inside a 3D physics simulator ([PyBullet](https://pybullet.org/wordpress/)). You can watch it move in real time, tweak sliders to change how tightly drones flock together, and see the hierarchy visualised as coloured lines.

---

## 2. The Three Tiers

Think of the swarm like a military unit with three levels of command.

```
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1 — GENERAL  (1 drone, purple)                            │
│  "The mission planner."                                          │
│  Divides the arena into zones, decides which zone needs work,    │
│  broadcasts anonymous goal tokens, watches zone health.         │
└────────────────────┬─────────────────────────────────────────────┘
                     │  GoalToken  (broadcast to all zones)
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  NODE     │ │  NODE     │ │  NODE     │  (one per zone, teal)
│  "The     │ │           │ │           │
│  zone     │ │           │ │           │
│  boss."   │ └──┬────┬───┘ └───────────┘
└──┬────┬───┘   ...  ...
   │    │
   │    └──────────────────────────────────────┐
   │    VelocityCommand                        │  TaskCommand
   ▼                                           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────┐
│  SCOUT   │ │  SCOUT   │ │  SCOUT   │   │  WORKER  │  (amber / blue)
│  "Eyes." │ │          │ │          │   │  "Hands."│
└──────────┘ └──────────┘ └──────────┘   └──────────┘
```

### General (Tier 1)
One drone, flies a little higher than everyone else. Its job is strategic: it keeps a **world model** — a table of every zone, its health, coverage, and whether the drones inside have gone silent. It cycles through mission phases (Explore → Converge → Execute → Withdraw → Rebalance) on a timer, and at each broadcast interval it emits a **GoalToken** for every active zone. GoalTokens tell Nodes what waypoint to aim for and what flight mode to use.

### Node (Tier 2)
One Node per zone. It collects sensor readings from its Scouts and task reports from its Workers, runs a neural attention network to compress them into a compact summary, and uses **Reynolds flocking rules** to decide where the zone's drones should be. It then broadcasts VelocityCommands (to Scouts) and TaskCommands (to Workers).

### Scout (Tier 3 — sensing)
Four Scouts per Node by default. Each Scout carries simulated sensors: a distance sensor for obstacles, an optical flow sensor for ground-relative speed, an inertial unit, and a barometer. Every tick it emits an anonymous **ScoutPacket** containing relative positions of nearby drones, sensor readings, and battery level — but never its own name or hardware ID.

### Worker (Tier 3 — actuation)
One Worker per Node by default. Workers execute physical tasks — moving to a target position and operating a payload (gripper, dispenser, or relay). A 5-state FSM (IDLE → MOVING → EXECUTING → COMPLETE / FAILED) tracks task progress. Workers also send anonymous status packets upward so the Node knows whether a task succeeded.

---

## 3. Why Drones Don't Have Names

Traditional swarm systems use hardware addresses to route commands ("drone 7, go left"). That creates a privacy problem: an eavesdropper who captures enough packets can figure out which physical drone is which, track individuals, and eventually map the whole fleet.

ASCS_3D takes a different approach — **zone-hash identity**. A drone's "name" is just the integer ID of the zone it currently belongs to (`row × grid_cols + col`). That integer is shared by every drone in the zone. Commands are broadcast to a zone, not to an individual. Sensor packets contain no source address.

This means:
- Captured packets reveal which zone sent data, not which physical device.
- You cannot reconstruct individual drone trajectories from packets alone.
- Replacing a broken drone is trivial — a new drone joins the zone and immediately has the same "name" as the one it replaced.

One consequence: the system cannot sort packets by sender and then process them in order — that would accidentally recreate individual identities via timing. The neural aggregator inside the Node is deliberately designed to be **permutation-invariant**, meaning the result is the same regardless of the order packets arrive in.

---

## 4. How the Swarm Moves — Reynolds Flocking

Nodes direct their Scouts using four simple rules borrowed from Craig Reynolds' 1987 model of bird flocks:

| Rule | What it does | Analogy |
|---|---|---|
| **Separation** | Push away from drones that are too close | Don't bump into your neighbour |
| **Alignment** | Match the average velocity of nearby drones | Fly the same direction as the flock |
| **Cohesion** | Steer toward the average position of nearby drones | Stay with the group |
| **Waypoint pull** | Steer toward the current goal waypoint | Follow the destination |

Each rule produces a velocity vector. The four vectors are added together with weights (`w_sep`, `w_align`, `w_coh`, `w_wp`). The sliders in the simulator let you adjust these weights live.

```
final_velocity = w_sep × v_sep
               + w_align × v_align
               + w_coh × v_coh
               + w_wp × v_waypoint
```

The weights depend on the current mission phase — during Explore the waypoint pull is stronger, during Execute separation is stronger to avoid interfering with the Worker.

**Reinforcement learning adjustment.** The Node contains a table of baseline weights per phase. An RL policy (not included in this simulation, but wired in via `set_rl_offsets()`) can nudge those weights up or down by at most ±0.25, keeping the drone safe while allowing learned improvements.

---

## 5. How It Scales

The arena is divided into a rectangular grid. The default is 2×2 (four zones). To run a 3×3 arena with nine Nodes:

```python
# In swarm_sim.py, change:
SWARM_CONFIG = {
    'arena_w': 30.0,
    'arena_h': 30.0,
    'grid_cols': 3,
    'grid_rows': 3,
    ...
}
```

The system handles more dynamic situations too:

- **Zone split:** If a zone becomes crowded (too many drones, too much collision risk, or too large a bounding box), it splits along its longer axis into two smaller zones. Each gets a new synthetic zone-hash.
- **Zone merge:** If two adjacent zones have been quiet for a while (low drone density, low collision risk), they merge into one larger zone to reduce overhead.

Splits and merges are handled automatically by the `ZoneMap` class. The General monitors conditions every tick and triggers splits/merges as needed. Spawning or despawning a Node is as simple as calling `SwarmController.spawn_node(zone_hash)` or `despawn_node(zone_hash)`.

---

## 6. Running It Yourself

### Prerequisites

```bash
pip install pybullet numpy
```

Python 3.9 or later. No GPU required.

### Launch

```bash
cd /path/to/ASCS_3D
python swarm_sim.py
```

A PyBullet physics window will open. The simulation starts immediately.

### What You See

| Colour | Agent |
|---|---|
| Purple sphere | General (flies highest) |
| Teal spheres | Nodes (one per zone) |
| Amber spheres | Scouts (four per Node) |
| Blue spheres | Workers (one per Node) |
| Teal lines | Node → Scout hierarchy |
| Blue lines | Node → Worker hierarchy |
| Flat cylinders on ground | Waypoint markers |

### The Sliders

Four sliders on the left of the window control Reynolds weights live:

- **w_sep** (0–1.5): Separation strength. Increase to spread drones out. Default 0.6.
- **w_align** (0–1.5): Alignment strength. Increase to make the flock fly more uniformly. Default 0.4.
- **w_coh** (0–1.5): Cohesion strength. Increase to pull drones tighter together. Default 0.5.
- **w_wp** (0–1.0): Waypoint pull. Increase to make Nodes chase their waypoint more aggressively. Default 0.35.

Changes take effect on the next simulation tick.

### Stopping

Close the PyBullet window or press Ctrl+C in the terminal. The simulation exits cleanly.

---

## 7. File Map

```
ASCS_3D/
│
├── General/
│   ├── __init__.py          — Package marker (empty)
│   ├── ZoneMap.py           — Arena geometry: zone grid, splits, merges, proximity
│   └── General.py           — Top-tier agent: world model, mission FSM, token broadcast
│
├── Node/
│   ├── __init__.py          — Package marker (empty)
│   └── Node.py              — Mid-tier agent: attention aggregator, Reynolds flocking, commands
│
├── Scout/
│   ├── __init__.py          — Package marker (empty)
│   └── Scout.py             — Sensing drone: virtual sensors, PID flight, packet emission
│
├── Worker/
│   ├── __init__.py          — Package marker (empty)
│   └── Worker.py            — Actuation drone: task FSM, payload operations, PID flight
│
├── controllers/
│   ├── __init__.py          — Package marker (empty)
│   ├── base_controller.py   — Abstract base: silence timers, uptime, event log, validation
│   ├── general_controller.py — Wraps General; drives FSM + zone checks each tick
│   ├── node_controller.py   — Wraps Node; orchestrates Scout/Worker sub-controllers
│   ├── scout_controller.py  — Wraps Scout; drives position + silence detection each tick
│   ├── worker_controller.py — Wraps Worker; drives task FSM + battery check each tick
│   └── swarm_controller.py  — Top-level: owns all controllers, runs the main step() loop
│
├── swarm_sim.py             — PyBullet visualiser: renders agents, sliders, hierarchy lines
│
└── docs/
    ├── TECHNICAL_REFERENCE.md — Full developer reference for all classes and protocols
    └── SYSTEM_OVERVIEW.md     — This file
```

---

*ASCS_3D is a research simulation. The anonymity guarantees, Reynolds weights, and FSM timers are tuned for a small arena. Real deployments would require hardware-in-the-loop testing, authenticated comms channels, and regulatory compliance.*
