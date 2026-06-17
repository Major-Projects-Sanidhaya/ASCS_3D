"""
ursina_renderer.py
------------------
3D visualization for ASCS_3D swarm using Ursina engine.

CRITICAL: This renderer is READ-ONLY. It NEVER modifies the swarm.
All data comes from RenderState snapshots.
"""

from ursina import *
import sys
import math


def make_drone_model(tier, base_color):
    """
    Build a quadcopter model using only built-in Ursina shapes.
    Uses cubes and spheres to create recognizable drone shape.

    Args:
        tier: 'GENERAL', 'NODE', 'SCOUT', or 'WORKER'
        base_color: RGB tuple for drone color

    Returns:
        Entity with quadcopter geometry as children
    """
    # Scale based on tier
    if tier == 'GENERAL':
        scale_factor = 1.5
    elif tier == 'NODE':
        scale_factor = 1.0
    elif tier == 'SCOUT':
        scale_factor = 0.6
    else:  # WORKER
        scale_factor = 0.7

    # Root entity
    root = Entity()

    # Central body (main cube)
    body = Entity(
        parent=root,
        model='cube',
        scale=(0.6 * scale_factor, 0.25 * scale_factor, 0.6 * scale_factor),
        color=rgb(*base_color),
        unlit=True,
    )

    # 4 arms in X configuration (thin cubes instead of cylinders)
    arm_positions = [
        ( 1.0,  1.0, 45),   # Front-right
        (-1.0,  1.0, -45),  # Front-left
        (-1.0, -1.0, 45),   # Back-left
        ( 1.0, -1.0, -45),  # Back-right
    ]

    for x, z, rot_y in arm_positions:
        # Arm (thin elongated cube)
        arm = Entity(
            parent=root,
            model='cube',
            scale=(0.7 * scale_factor, 0.08 * scale_factor, 0.12 * scale_factor),
            position=(x * scale_factor * 0.35, 0, z * scale_factor * 0.35),
            rotation=(0, rot_y, 0),
            color=color.dark_gray,
            unlit=True,
        )

        # Rotor (flat cube at arm end)
        rotor = Entity(
            parent=root,
            model='cube',
            scale=(0.35 * scale_factor, 0.03 * scale_factor, 0.35 * scale_factor),
            position=(x * scale_factor * 0.7, 0.15 * scale_factor, z * scale_factor * 0.7),
            color=color.light_gray,
            unlit=True,
        )

        # Rotor center dot
        rotor_center = Entity(
            parent=root,
            model='sphere',
            scale=0.08 * scale_factor,
            position=(x * scale_factor * 0.7, 0.15 * scale_factor, z * scale_factor * 0.7),
            color=color.black,
            unlit=True,
        )

    # Camera/sensor (small sphere on top)
    camera = Entity(
        parent=root,
        model='sphere',
        scale=0.15 * scale_factor,
        position=(0, 0.2 * scale_factor, 0.3 * scale_factor),
        color=color.black,
        unlit=True,
    )

    # Worker: add red suppression tank underneath
    if tier == 'WORKER':
        tank = Entity(
            parent=root,
            model='cube',
            scale=(0.4 * scale_factor, 0.3 * scale_factor, 0.4 * scale_factor),
            position=(0, -0.3 * scale_factor, 0),
            color=color.red,
            unlit=True,
        )

    return root


class UrsinaRenderer:
    """
    3D renderer for ASCS_3D swarm visualization.

    Consumes RenderState snapshots - never modifies simulation.
    """

    def __init__(self, swarm_controller, scenario='house_fire'):
        """
        Initialize Ursina renderer.

        Args:
            swarm_controller: SwarmController instance (READ ONLY)
            scenario: Scenario name for display
        """
        self.swarm = swarm_controller
        self.scenario = scenario

        # Initialize Ursina app
        self.app = Ursina(title=f'ASCS_3D - {scenario}', borderless=False)

        # Set sky/background color (lighter than default black)
        window.color = color.rgb(30, 30, 40)  # Dark blue-gray

        # Drone entities (keyed by drone_id)
        self.drone_entities = {}

        # Zone entities (keyed by zone_hash)
        self.zone_entities = {}
        self.human_markers = {}
        self.fire_entities = {}  # Fire visuals (keyed by zone_hash)
        self.fire_lights = {}    # Fire point lights (keyed by zone_hash)

        # Create ground plane
        self._create_ground()

        # Create camera
        self._create_camera()

        # Create UI elements
        self._create_ui()

        # Frame counter
        self.frame_count = 0

        # Create an invisible entity with update method
        # Ursina calls update() on all entities automatically
        class UpdateHandler(Entity):
            def __init__(self, renderer):
                super().__init__()
                self.renderer = renderer

            def update(self):
                self.renderer.update()

        self.update_handler = UpdateHandler(self)

        print(f"[Ursina] Renderer initialized for scenario: {scenario}")
        print(f"[Ursina] Camera position: {camera.position}")
        print(f"[Ursina] Camera rotation: {camera.rotation}")
        print(f"[Ursina] Update handler registered")

    def _create_ground(self):
        """Create ground plane with grid."""
        # ADD LIGHTING - Critical for seeing 3D objects!
        DirectionalLight(
            y=10,
            z=-10,
            rotation=(45, -45, 0),
            shadows=False,
        )

        # Ambient light for fill
        AmbientLight(color=color.rgba(100, 100, 100, 0.5))

        # Sky sphere for depth perception
        Sky(color=color.rgb(135, 206, 235))  # Light blue sky

        # Main ground plane
        self.ground = Entity(
            model='plane',
            scale=(40, 1, 40),  # 40x40m
            texture='white_cube',
            color=color.gray,
            collider='box',
        )

        # Grid lines
        for i in range(-20, 21, 5):
            # X-axis lines
            Entity(
                model='cube',
                scale=(40, 0.01, 0.1),
                position=(0, 0.01, i),
                color=color.dark_gray,
            )
            # Z-axis lines
            Entity(
                model='cube',
                scale=(0.1, 0.01, 40),
                position=(i, 0.01, 0),
                color=color.dark_gray,
            )

    def _create_camera(self):
        """Create orbital camera."""
        # Enable orbital camera with mouse FIRST
        self.editor_camera = EditorCamera()
        self.editor_camera.rotation_speed = 200
        self.editor_camera.pan_speed = (10, 10, 10)

        # THEN set camera position/rotation (after EditorCamera created)
        camera.position = (0, 25, -25)  # Higher up, looking down
        camera.rotation_x = 45  # Look down at ground
        camera.fov = 60  # Field of view

    def _create_ui(self):
        """Create HUD text elements."""
        # LLM feed (top-left)
        self.llm_text = Text(
            text='LLM Feed:\n',
            position=(-0.85, 0.45),
            scale=0.8,
            origin=(0, 0),
            background=True,
        )

        # Status HUD (bottom)
        self.status_text = Text(
            text='Mission Status',
            position=(0, -0.45),
            scale=1.0,
            origin=(0, 0),
            background=True,
        )

        # FPS counter (top-right)
        self.fps_text = Text(
            text='FPS: 0',
            position=(0.7, 0.45),
            scale=0.8,
            origin=(0, 0),
        )

    def update(self):
        """
        Update loop - called every frame by Ursina.

        Reads RenderState and updates entity positions.
        CRITICAL: Never modifies swarm - read only via get_render_state().
        """
        # Debug: Print first few frames
        if self.frame_count < 5:
            print(f"[Ursina] UPDATE called - Frame {self.frame_count}")

        # Step the simulation
        self.swarm.step(1.0 / 60.0)  # 60 FPS target

        # Get render state (pure data snapshot)
        render_state = self.swarm.get_render_state()

        # Debug: Print render state info
        if self.frame_count < 5:
            print(f"[Ursina] RenderState: {len(render_state.drones)} drones, {len(render_state.zones)} zones")

        # Update drones
        self._update_drones(render_state.drones)

        # Update zones
        self._update_zones(render_state.zones)

        # Update UI
        self._update_ui(render_state)

        # Frame counter
        self.frame_count += 1

        # Update FPS (handle first frame where dt may be 0)
        if self.frame_count % 10 == 0 and time.dt > 0:
            fps = int(1.0 / time.dt)
            self.fps_text.text = f'FPS: {fps}'
        elif self.frame_count == 1:
            self.fps_text.text = 'FPS: Starting...'

    def _update_drones(self, drones):
        """Update drone entities from render state."""
        # Track which drones we've seen
        seen_ids = set()

        # Debug: Print drone count on first few frames
        if self.frame_count < 3:
            print(f"[Ursina] Frame {self.frame_count}: Rendering {len(drones)} drones")

        for drone in drones:
            seen_ids.add(drone.drone_id)

            # Create entity if new
            if drone.drone_id not in self.drone_entities:
                # Create quadcopter model
                entity = make_drone_model(drone.tier, drone.color)
                entity.position = drone.position
                self.drone_entities[drone.drone_id] = entity

                # Debug: Print first few entity creations
                if len(self.drone_entities) <= 5:
                    print(f"[Ursina] Created {drone.tier} quadcopter at {drone.position}")

            # Update position
            entity = self.drone_entities[drone.drone_id]
            entity.position = drone.position

        # Remove entities for drones that no longer exist
        for drone_id in list(self.drone_entities.keys()):
            if drone_id not in seen_ids:
                destroy(self.drone_entities[drone_id])
                del self.drone_entities[drone_id]

    def _update_zones(self, zones):
        """Update zone heatmap and human markers."""
        seen_hashes = set()

        for zone in zones:
            seen_hashes.add(zone.zone_hash)

            # Create zone heatmap quad if new
            if zone.zone_hash not in self.zone_entities:
                # Zone size from zone map
                zone_w = 20.0 / 2  # Default 2x2 grid = 10m per zone
                zone_h = 20.0 / 2

                quad = Entity(
                    model='plane',
                    scale=(zone_w, 1, zone_h),
                    position=(zone.center[0], 0.05, zone.center[1]),
                    color=color.blue,
                    alpha=0.3,
                )
                self.zone_entities[zone.zone_hash] = quad

            # Update color based on fire intensity
            quad = self.zone_entities[zone.zone_hash]
            # Blue (cool) to Red (hot)
            fire_color = color.rgb(
                zone.fire_intensity,  # R increases with fire
                0.2,  # G stays low
                1.0 - zone.fire_intensity  # B decreases with fire
            )
            quad.color = fire_color

            # Fire visuals (if fire_intensity > 0.1)
            if zone.fire_intensity > 0.1:
                if zone.zone_hash not in self.fire_entities:
                    # Create fire effect (stacked translucent quads)
                    fire_stack = Entity()

                    # 3 layers of flickering fire
                    for i in range(3):
                        fire_layer = Entity(
                            parent=fire_stack,
                            model='plane',
                            scale=(3, 3),
                            position=(zone.center[0], 0.5 + i * 0.8, zone.center[1]),
                            rotation_x=90,
                            color=color.rgb(255, 100 + i * 40, 0),  # Orange to red gradient
                            alpha=0.4,
                            unlit=True,
                        )

                    self.fire_entities[zone.zone_hash] = fire_stack

                    # Add point light
                    fire_light = PointLight(
                        position=(zone.center[0], 1.5, zone.center[1]),
                        color=color.rgb(255, 100, 0),
                    )
                    self.fire_lights[zone.zone_hash] = fire_light

                # Animate fire (flicker)
                fire_stack = self.fire_entities[zone.zone_hash]
                for i, child in enumerate(fire_stack.children):
                    # Vary alpha for flicker effect
                    flicker = 0.3 + 0.2 * math.sin(time.time() * 3 + i)
                    child.alpha = flicker * zone.fire_intensity
                    # Scale based on fire intensity
                    child.scale = (2 + zone.fire_intensity * 2, 2 + zone.fire_intensity * 2)

                # Update light intensity
                if zone.zone_hash in self.fire_lights:
                    self.fire_lights[zone.zone_hash].color = color.rgb(
                        255,
                        int(100 * zone.fire_intensity),
                        0
                    )
            else:
                # Remove fire visuals if fire suppressed
                if zone.zone_hash in self.fire_entities:
                    destroy(self.fire_entities[zone.zone_hash])
                    del self.fire_entities[zone.zone_hash]
                if zone.zone_hash in self.fire_lights:
                    destroy(self.fire_lights[zone.zone_hash])
                    del self.fire_lights[zone.zone_hash]

            # Human present marker (improved - blue beam + pulsing icon)
            if zone.human_present:
                if zone.zone_hash not in self.human_markers:
                    # Container for human marker
                    marker_container = Entity()

                    # Vertical blue beam (more visible than yellow)
                    beam = Entity(
                        parent=marker_container,
                        model='cylinder',
                        scale=(0.4, 6, 0.4),
                        position=(zone.center[0], 3, zone.center[1]),
                        color=color.cyan,
                        alpha=0.6,
                        unlit=True,
                    )

                    # Pulsing icon above zone
                    icon = Entity(
                        parent=marker_container,
                        model='sphere',
                        scale=0.8,
                        position=(zone.center[0], 6, zone.center[1]),
                        color=color.cyan,
                        unlit=True,
                    )

                    self.human_markers[zone.zone_hash] = marker_container

                # Animate pulse
                marker_container = self.human_markers[zone.zone_hash]
                if len(marker_container.children) >= 2:
                    icon = marker_container.children[1]
                    pulse = 0.8 + 0.3 * math.sin(time.time() * 4)
                    icon.scale = pulse
            else:
                # Remove marker if human no longer present
                if zone.zone_hash in self.human_markers:
                    destroy(self.human_markers[zone.zone_hash])
                    del self.human_markers[zone.zone_hash]

        # Clean up zones that no longer exist
        for zone_hash in list(self.zone_entities.keys()):
            if zone_hash not in seen_hashes:
                destroy(self.zone_entities[zone_hash])
                del self.zone_entities[zone_hash]

    def _update_ui(self, render_state):
        """Update UI text from render state."""
        # LLM Feed
        llm_text = 'LLM Feed:\n'
        for i, msg in enumerate(render_state.llm_messages[-5:]):
            llm_text += f'{i+1}. {msg[:60]}...\n'  # Truncate long messages
        self.llm_text.text = llm_text

        # Status HUD
        num_drones = len(render_state.drones)
        num_zones = len(render_state.zones)
        active_fires = sum(1 for z in render_state.zones if z.fire_intensity > 0.1)
        humans_detected = sum(1 for z in render_state.zones if z.human_present)

        status = f'Scenario: {self.scenario} | '
        status += f'Drones: {num_drones} | '
        status += f'Zones: {num_zones} | '
        status += f'Active Fires: {active_fires} | '
        status += f'Humans Detected: {humans_detected}'

        self.status_text.text = status

    def run(self):
        """Start the Ursina render loop."""
        print("[Ursina] Starting render loop...")
        print("[Ursina] Controls: Mouse drag to rotate, scroll to zoom, ESC to exit")

        # Clean exit handler
        def on_exit():
            print("[Ursina] Closing renderer...")
            sys.exit(0)

        self.app.on_destroy = on_exit

        # Run the app
        self.app.run()
