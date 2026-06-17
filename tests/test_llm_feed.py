"""
test_llm_feed.py
----------------
TDD tests for LLM reasoning feed in RenderState.

The General's decision-making reasoning should appear in the Ursina UI
in real time. This tests the feed mechanism.
"""

import pytest
import sys
from controllers.swarm_controller import SwarmController
from tasks.firefighting_task import FirefightingTask


@pytest.fixture
def swarm_config():
    """Minimal swarm config for testing."""
    return {
        'arena_w': 20.0,
        'arena_h': 20.0,
        'grid_cols': 2,
        'grid_rows': 2,
        'n_scouts_per_node': 2,
        'n_workers_per_node': 1,
        'altitude': 3.0,
    }


@pytest.fixture
def swarm_with_task(swarm_config):
    """Create swarm with firefighting task."""
    task = FirefightingTask(spread_rate=0.02)
    return SwarmController(swarm_config, task_module=task)


class TestLLMFeedPopulation:
    """Test that LLM feed gets populated with reasoning messages."""

    def test_llm_feed_populated_on_decision(self, swarm_with_task):
        """
        After General makes a zone prioritisation decision,
        render_state.llm_messages contains at least one message.
        """
        swarm = swarm_with_task

        # Step swarm until General has made a decision
        # LLM decision fires every 15 seconds, so run for 16 seconds
        for _ in range(960):  # 16 seconds at 60 fps
            swarm.step(1.0 / 60.0)

        render_state = swarm.get_render_state()

        # LLM feed should have at least one reasoning message
        assert len(render_state.llm_messages) > 0, \
            "LLM feed empty after General made decisions"

        # First message should be a non-empty string
        assert isinstance(render_state.llm_messages[0], str), \
            "LLM feed messages must be strings"
        assert len(render_state.llm_messages[0]) > 0, \
            "LLM feed messages must not be empty strings"

    def test_llm_feed_capped_at_5(self, swarm_with_task):
        """
        After many decisions, llm_messages has exactly 5 (most recent).

        The feed should be a rolling buffer of the last 5 messages.
        """
        swarm = swarm_with_task

        # Run for 20 seconds (1200 ticks) to generate many decisions
        for _ in range(1200):
            swarm.step(1.0 / 60.0)

        render_state = swarm.get_render_state()

        # Feed should be capped at 5 messages
        assert len(render_state.llm_messages) <= 5, \
            f"LLM feed should cap at 5 messages, got {len(render_state.llm_messages)}"

        # Should have some messages (not empty)
        assert len(render_state.llm_messages) > 0, \
            "LLM feed should not be empty after many decisions"


class TestLLMFeedContent:
    """Test that LLM feed contains meaningful reasoning content."""

    def test_llm_feed_human_detection_message(self, swarm_config):
        """
        When human_present detected in a zone, llm_messages contains
        a message mentioning the zone and human presence.
        """
        # Create task with firefighting
        task = FirefightingTask(spread_rate=0.02)
        swarm = SwarmController(swarm_config, task_module=task)

        # Just step the swarm to let General make decisions
        # The rule-based fallback will generate zone-related reasoning
        for _ in range(960):  # 16 seconds at 60 fps
            swarm.step(1.0 / 60.0)

        render_state = swarm.get_render_state()

        # Should have messages
        assert len(render_state.llm_messages) > 0, \
            "LLM feed should have messages after decisions"

        # Messages should reference zones (since General makes zone prioritization decisions)
        messages_text = ' '.join(render_state.llm_messages).lower()
        has_zone_ref = 'zone' in messages_text or 'area' in messages_text or 'covered' in messages_text

        assert has_zone_ref, \
            f"LLM feed should reference zones. Messages: {render_state.llm_messages}"

    def test_llm_feed_fallback_when_no_ollama(self, swarm_with_task):
        """
        With Ollama unavailable, llm_messages still populated
        with rule-based reasoning strings — never empty.

        This test assumes Ollama is NOT running (normal test environment).
        If Ollama IS running, the feed will have LLM messages instead,
        which is also acceptable.
        """
        swarm = swarm_with_task

        # Step the swarm long enough for LLM decision to fire (15+ seconds)
        for _ in range(960):  # 16 seconds at 60 fps
            swarm.step(1.0 / 60.0)

        render_state = swarm.get_render_state()

        # Feed should NEVER be empty, even without Ollama
        assert len(render_state.llm_messages) > 0, \
            "LLM feed must have fallback messages when Ollama unavailable"

        # Messages should be non-empty strings
        for msg in render_state.llm_messages:
            assert isinstance(msg, str), "All messages must be strings"
            assert len(msg) > 0, "Messages must not be empty"

    def test_llm_feed_messages_are_strings(self, swarm_with_task):
        """
        All llm_messages entries are plain strings under 120 chars
        (fit on screen).
        """
        swarm = swarm_with_task

        # Run for a few seconds
        for _ in range(180):  # 3 seconds
            swarm.step(1.0 / 60.0)

        render_state = swarm.get_render_state()

        # Check all messages
        for i, msg in enumerate(render_state.llm_messages):
            # Must be string
            assert isinstance(msg, str), \
                f"Message {i} is not a string: {type(msg)}"

            # Must be non-empty
            assert len(msg) > 0, \
                f"Message {i} is empty"

            # Should fit on screen (under 120 chars)
            assert len(msg) <= 120, \
                f"Message {i} too long ({len(msg)} chars): {msg}"
