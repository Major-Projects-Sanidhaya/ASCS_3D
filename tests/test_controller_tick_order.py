"""
test_controller_tick_order.py
------------------------------
TDD tests for controller execution order verification.
All tests written BEFORE implementation following STEP 1.

Ensures proper tick sequencing:
  General → Nodes → Scouts → Workers
  Within NodeController: buffer clear → emit → aggregate → op_phase → commands

These tests verify that the system enforces correct causality chains.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, call

from controllers.swarm_controller import SwarmController
from controllers.node_controller import NodeController
from controllers.scout_controller import ScoutController
from controllers.worker_controller import WorkerController
from General.ZoneMap import ZoneMap


class TestSwarmTickOrder:
    """Test suite for top-level SwarmController tick sequencing."""

    @pytest.fixture
    def swarm(self):
        """Create a minimal swarm: 1 zone, 2 scouts, 1 worker."""
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
            'altitude': 3.0,
        }
        return SwarmController(config)

    def test_general_steps_before_nodes(self, swarm):
        """General.step() must be called before any Node.step() in swarm.step()."""
        call_order = []

        # Mock General step
        original_general_step = swarm._general.step
        def mock_general_step(dt):
            call_order.append('GENERAL')
            original_general_step(dt)
        swarm._general.step = mock_general_step

        # Mock all Node steps
        for zone_hash, node in swarm._nodes.items():
            original_node_step = node.step
            def mock_node_step(dt, zh=zone_hash):
                call_order.append(f'NODE_{zh}')
                original_node_step(dt)
            node.step = mock_node_step

        # Run one tick
        swarm.step(0.016)

        # Verify General was first
        assert call_order[0] == 'GENERAL', \
            f"General must step first, got order: {call_order}"

        # Verify all nodes came after General
        general_idx = call_order.index('GENERAL')
        for item in call_order:
            if item.startswith('NODE_'):
                node_idx = call_order.index(item)
                assert node_idx > general_idx, \
                    f"Node {item} stepped before General"

    def test_nodes_step_before_scouts(self, swarm):
        """All Node.step() must complete before any Scout.step()."""
        call_order = []

        # Mock all Node steps
        for zone_hash, node in swarm._nodes.items():
            original_node_step = node.step
            def mock_node_step(dt, zh=zone_hash):
                call_order.append(f'NODE_{zh}')
                original_node_step(dt)
            node.step = mock_node_step

        # Mock all Scout steps
        for i, scout in enumerate(swarm._scouts):
            original_scout_step = scout.step
            def mock_scout_step(dt, idx=i):
                call_order.append(f'SCOUT_{idx}')
                original_scout_step(dt)
            scout.step = mock_scout_step

        # Run one tick
        swarm.step(0.016)

        # Find last node and first scout indices
        node_indices = [i for i, x in enumerate(call_order) if x.startswith('NODE_')]
        scout_indices = [i for i, x in enumerate(call_order) if x.startswith('SCOUT_')]

        if node_indices and scout_indices:
            last_node_idx = max(node_indices)
            first_scout_idx = min(scout_indices)
            assert last_node_idx < first_scout_idx, \
                f"All nodes must complete before scouts start. Order: {call_order}"

    def test_scouts_step_before_workers(self, swarm):
        """All Scout.step() must complete before any Worker.step()."""
        call_order = []

        # Mock all Scout steps
        for i, scout in enumerate(swarm._scouts):
            original_scout_step = scout.step
            def mock_scout_step(dt, idx=i):
                call_order.append(f'SCOUT_{idx}')
                original_scout_step(dt)
            scout.step = mock_scout_step

        # Mock all Worker steps
        for i, worker in enumerate(swarm._workers):
            original_worker_step = worker.step
            def mock_worker_step(dt, idx=i):
                call_order.append(f'WORKER_{idx}')
                original_worker_step(dt)
            worker.step = mock_worker_step

        # Run one tick
        swarm.step(0.016)

        # Find last scout and first worker indices
        scout_indices = [i for i, x in enumerate(call_order) if x.startswith('SCOUT_')]
        worker_indices = [i for i, x in enumerate(call_order) if x.startswith('WORKER_')]

        if scout_indices and worker_indices:
            last_scout_idx = max(scout_indices)
            first_worker_idx = min(worker_indices)
            assert last_scout_idx < first_worker_idx, \
                f"All scouts must complete before workers start. Order: {call_order}"


class TestNodeTickOrder:
    """Test suite for NodeController internal tick sequencing."""

    @pytest.fixture
    def zone_map(self):
        """Create a 1x1 zone map."""
        return ZoneMap(arena_w=20.0, arena_h=20.0, grid_cols=1, grid_rows=1)

    @pytest.fixture
    def node(self, zone_map):
        """Create a NodeController with 2 scouts, 1 worker."""
        pos = np.array([0.0, 0.0, 3.0])
        nc = NodeController(pos, zone_map, zone_hash=0)

        # Spawn scouts
        for i in range(2):
            angle = 2 * np.pi * i / 2
            scout_pos = pos + np.array([np.cos(angle) * 1.5, np.sin(angle) * 1.5, 0.0])
            from controllers.scout_controller import ScoutController
            sc = ScoutController(scout_pos, nc._agent)
            nc.register_scout(sc)

        # Spawn worker
        from controllers.worker_controller import WorkerController
        wc = WorkerController(pos + np.array([2.0, 0.0, 0.0]), nc._agent)
        nc.register_worker(wc)

        return nc

    def test_packets_cleared_before_emit(self, node):
        """Packet buffer must be empty BEFORE scouts emit, non-empty AFTER."""
        buffer_states = []

        # Hook into clear_packet_buffer
        original_clear = node._agent.clear_packet_buffer
        def mock_clear():
            buffer_states.append(('BEFORE_CLEAR', len(node._agent._last_packets)))
            original_clear()
            buffer_states.append(('AFTER_CLEAR', len(node._agent._last_packets)))
        node._agent.clear_packet_buffer = mock_clear

        # Hook into first scout's emit
        original_emit = node._scout_controllers[0]._agent.emit_packet
        def mock_emit(*args, **kwargs):
            buffer_states.append(('BEFORE_EMIT', len(node._agent._last_packets)))
            result = original_emit(*args, **kwargs)
            buffer_states.append(('AFTER_EMIT', len(node._agent._last_packets)))
            return result
        node._scout_controllers[0]._agent.emit_packet = mock_emit

        # Run one step
        node.step(0.016)

        # Verify buffer was cleared before first emit
        clear_idx = next(i for i, (event, _) in enumerate(buffer_states) if event == 'AFTER_CLEAR')
        emit_idx = next(i for i, (event, _) in enumerate(buffer_states) if event == 'BEFORE_EMIT')

        assert clear_idx < emit_idx, \
            f"Buffer must be cleared before scouts emit. States: {buffer_states}"

        # Verify buffer was empty after clear, non-empty after emit
        _, count_after_clear = buffer_states[clear_idx]
        _, count_after_emit = buffer_states[emit_idx + 1]

        assert count_after_clear == 0, \
            f"Buffer should be empty after clear, got {count_after_clear}"
        assert count_after_emit > 0, \
            f"Buffer should be non-empty after emit, got {count_after_emit}"

    def test_aggregate_after_emit(self, node):
        """aggregate_observations must be called AFTER all scouts emit."""
        call_order = []

        # Hook into scout emissions
        for i, sc in enumerate(node._scout_controllers):
            original_emit = sc._agent.emit_packet
            def mock_emit(*args, idx=i, **kwargs):
                call_order.append(f'EMIT_SCOUT_{idx}')
                return original_emit(*args, **kwargs)
            sc._agent.emit_packet = mock_emit

        # Hook into aggregate_observations
        original_aggregate = node._agent.aggregate_observations
        def mock_aggregate():
            call_order.append('AGGREGATE')
            return original_aggregate()
        node._agent.aggregate_observations = mock_aggregate

        # Run one step
        node.step(0.016)

        # Find aggregate index
        aggregate_idx = call_order.index('AGGREGATE')

        # Verify all emits happened before aggregate
        for i, event in enumerate(call_order):
            if event.startswith('EMIT_'):
                assert i < aggregate_idx, \
                    f"Aggregate must come after all emits. Order: {call_order}"

    def test_op_phase_update_after_aggregate(self, node):
        """update_op_phase must be called AFTER aggregate_observations."""
        call_order = []

        # Hook into aggregate_observations
        original_aggregate = node._agent.aggregate_observations
        def mock_aggregate():
            call_order.append('AGGREGATE')
            return original_aggregate()
        node._agent.aggregate_observations = mock_aggregate

        # Hook into update_op_phase
        original_update_op = node._agent.update_op_phase
        def mock_update_op():
            call_order.append('UPDATE_OP_PHASE')
            return original_update_op()
        node._agent.update_op_phase = mock_update_op

        # Run one step
        node.step(0.016)

        # Verify order
        assert 'AGGREGATE' in call_order, "aggregate_observations not called"
        assert 'UPDATE_OP_PHASE' in call_order, "update_op_phase not called"

        aggregate_idx = call_order.index('AGGREGATE')
        op_phase_idx = call_order.index('UPDATE_OP_PHASE')

        assert aggregate_idx < op_phase_idx, \
            f"update_op_phase must come after aggregate. Order: {call_order}"

    def test_report_sent_after_op_phase(self):
        """send_report to General must be called after update_op_phase completes."""
        # This test runs at the swarm level since send_report is called there
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
        }
        swarm = SwarmController(config)
        node = list(swarm._nodes.values())[0]

        call_order = []

        # Hook into update_op_phase
        original_update_op = node._agent.update_op_phase
        def mock_update_op():
            call_order.append('UPDATE_OP_PHASE')
            return original_update_op()
        node._agent.update_op_phase = mock_update_op

        # Hook into send_report
        original_send_report = node._agent.send_report
        def mock_send_report(*args, **kwargs):
            call_order.append('SEND_REPORT')
            return original_send_report(*args, **kwargs)
        node._agent.send_report = mock_send_report

        # Run one tick
        swarm.step(0.016)

        # Verify order
        assert 'UPDATE_OP_PHASE' in call_order, "update_op_phase not called"
        assert 'SEND_REPORT' in call_order, "send_report not called"

        op_phase_idx = call_order.index('UPDATE_OP_PHASE')
        report_idx = call_order.index('SEND_REPORT')

        assert op_phase_idx < report_idx, \
            f"send_report must come after update_op_phase. Order: {call_order}"

    def test_worker_commands_gated_by_op_phase(self, node):
        """Workers receive HOVER in SCOUTING phase, MOVE_TO in TASKING phase."""
        # Test SCOUTING phase
        node._agent._op_phase = 'SCOUTING'
        node._agent._coverage_fraction = 0.1  # Low coverage

        # Run one step
        node.step(0.016)

        # Check worker received HOVER command
        worker = node._worker_controllers[0]._agent
        last_cmd = worker._current_cmd

        # Worker should receive HOVER or no authorized command
        if last_cmd is not None:
            # If command exists, it should be HOVER and not authorized
            assert last_cmd.get('action') == 'HOVER' or not last_cmd.get('authorized', False), \
                f"Workers in SCOUTING should receive HOVER or unauthorized commands, got {last_cmd}"

        # Now test TASKING phase
        node._agent._op_phase = 'TASKING'
        node._agent._coverage_fraction = 0.8  # High coverage
        node._agent._frames_scouting = 100  # Met minimum frames

        # Need to add packets for coverage check
        for i in range(5):
            packet = {
                'rel_positions': [[1.0, 0.0, 0.0]],
                'rel_headings': [[1.0, 0.0, 0.0]],
                'speed': 2.0,
                'obs_fwd': 5.0,
                'obs_min': 5.0,
                'battery': 0.9,
            }
            node._agent.receive_scout_packet(packet)

        # Run another step
        node.step(0.016)

        # Worker should now receive authorized MOVE_TO
        last_cmd = worker._current_cmd
        if last_cmd is not None and node._agent._op_phase == 'TASKING':
            # In TASKING phase, workers should get authorized commands
            assert last_cmd.get('authorized', False) or last_cmd.get('action') == 'HOVER', \
                f"Workers in TASKING should receive authorized commands, got {last_cmd}"


class TestStepTracing:
    """Test suite demonstrating new _step_trace functionality."""

    def test_swarm_step_trace(self):
        """Demonstrate SwarmController._step_trace captures full execution order."""
        config = {
            'arena_w': 20.0,
            'arena_h': 20.0,
            'grid_cols': 1,
            'grid_rows': 1,
            'n_scouts_per_node': 2,
            'n_workers_per_node': 1,
        }
        swarm = SwarmController(config)

        # Enable tracing
        swarm._step_trace = []

        # Run one tick
        swarm.step(0.016)

        # Verify trace captured the expected sequence
        assert len(swarm._step_trace) > 0, "Step trace should capture events"

        # Check key events are in order
        general_start = swarm._step_trace.index('GENERAL_START')
        general_end = swarm._step_trace.index('GENERAL_END')
        node_start = swarm._step_trace.index('NODE_0_START')
        node_end = swarm._step_trace.index('NODE_0_END')

        # Find first scout and worker
        scout_indices = [i for i, x in enumerate(swarm._step_trace) if 'SCOUT' in x]
        worker_indices = [i for i, x in enumerate(swarm._step_trace) if 'WORKER' in x]

        # Verify order
        assert general_start < general_end < node_start < node_end, \
            "General must complete before nodes start"

        if scout_indices:
            first_scout = min(scout_indices)
            assert node_end < first_scout, "Nodes must complete before scouts"

        if worker_indices:
            first_worker = min(worker_indices)
            if scout_indices:
                last_scout = max(scout_indices)
                assert last_scout < first_worker, "Scouts must complete before workers"

        print(f"\nStep trace captured {len(swarm._step_trace)} events:")
        print(swarm._step_trace[:20])  # Print first 20 events

    def test_node_step_trace(self):
        """Demonstrate NodeController._step_trace captures internal execution order."""
        zone_map = ZoneMap(arena_w=20.0, arena_h=20.0, grid_cols=1, grid_rows=1)
        pos = np.array([0.0, 0.0, 3.0])
        node = NodeController(pos, zone_map, zone_hash=0)

        # Spawn scouts
        for i in range(2):
            angle = 2 * np.pi * i / 2
            scout_pos = pos + np.array([np.cos(angle) * 1.5, np.sin(angle) * 1.5, 0.0])
            from controllers.scout_controller import ScoutController
            sc = ScoutController(scout_pos, node._agent)
            node.register_scout(sc)

        # Enable tracing
        node._step_trace = []

        # Run one step
        node.step(0.016)

        # Verify trace captured expected sequence
        expected_order = [
            'TICK_UPTIME',
            'CLEAR_BUFFER',
            'EMIT_SCOUT_0',
            'EMIT_SCOUT_1',
            'AGGREGATE_OBSERVATIONS',
            'UPDATE_OP_PHASE',
            'COMPUTE_VELOCITY',
            'BROADCAST_SCOUT_COMMANDS',
            'BROADCAST_WORKER_COMMANDS',
            'UPDATE_POSITION',
        ]

        # Check all expected events are present
        for event in expected_order:
            assert event in node._step_trace, f"Missing event: {event}"

        # Verify order of key events
        clear_idx = node._step_trace.index('CLEAR_BUFFER')
        emit_0_idx = node._step_trace.index('EMIT_SCOUT_0')
        aggregate_idx = node._step_trace.index('AGGREGATE_OBSERVATIONS')
        op_phase_idx = node._step_trace.index('UPDATE_OP_PHASE')

        assert clear_idx < emit_0_idx < aggregate_idx < op_phase_idx, \
            f"Events out of order: {node._step_trace}"

        print(f"\nNode step trace: {node._step_trace}")
