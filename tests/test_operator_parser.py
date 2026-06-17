"""
test_operator_parser.py
-----------------------
TDD tests for operator instruction parsing.
All tests written BEFORE implementation.

OperatorParser converts natural language instructions into TaskCommands
with LLM-based parsing and rule-based fallback.
"""

import pytest

from tasks.operator_parser import OperatorParser, ClarificationRequest
from tasks.contracts import TaskCommand, ZoneThreatAssessment


class TestParserBasics:
    """Test suite for basic parsing functionality."""

    @pytest.mark.fast
    def test_parse_returns_task_command_or_clarification(self):
        """
        parse_operator_instruction always returns either TaskCommand or ClarificationRequest.
        Never raises exception, even with nonsense input like "asdfgh".
        """
        parser = OperatorParser()

        # Test with nonsense input
        result = parser.parse_operator_instruction("asdfgh", world_model={})

        assert isinstance(result, (TaskCommand, ClarificationRequest)), \
            f"Parser must return TaskCommand or ClarificationRequest, got {type(result)}"

        # Test with empty input
        result_empty = parser.parse_operator_instruction("", world_model={})
        assert isinstance(result_empty, (TaskCommand, ClarificationRequest))

        # Test with gibberish
        result_gibberish = parser.parse_operator_instruction("xyzabc123!@#", world_model={})
        assert isinstance(result_gibberish, (TaskCommand, ClarificationRequest))


class TestBasicCommands:
    """Test suite for parsing basic operator commands."""

    @pytest.mark.fast
    def test_parse_send_worker_to_zone(self):
        """
        Input: "send worker to zone 2"
        Returns TaskCommand with zone_hash=2, action_type=MOVE_TO
        """
        parser = OperatorParser()

        result = parser.parse_operator_instruction("send worker to zone 2", world_model={})

        assert isinstance(result, TaskCommand), \
            f"Should return TaskCommand for clear instruction, got {type(result)}"
        assert result.zone_hash == 2
        assert result.action_type == "MOVE_TO", \
            f"Send worker should map to MOVE_TO, got {result.action_type}"

    @pytest.mark.fast
    def test_parse_suppress_fire_zone(self):
        """
        Input: "suppress the fire in zone 3"
        Returns TaskCommand with zone_hash=3, action_type=SUPPRESS
        """
        parser = OperatorParser()

        result = parser.parse_operator_instruction("suppress the fire in zone 3", world_model={})

        assert isinstance(result, TaskCommand)
        assert result.zone_hash == 3
        assert result.action_type == "SUPPRESS"

    @pytest.mark.fast
    def test_parse_evacuate_person(self):
        """
        Input: "there is a child in zone 1, evacuate"
        Returns TaskCommand with zone_hash=1, action_type=EVACUATE, priority >= 8
        """
        parser = OperatorParser()

        result = parser.parse_operator_instruction(
            "there is a child in zone 1, evacuate",
            world_model={}
        )

        assert isinstance(result, TaskCommand)
        assert result.zone_hash == 1
        assert result.action_type == "EVACUATE"
        assert result.priority >= 8, \
            f"Evacuate child should be high priority, got {result.priority}"


class TestValidation:
    """Test suite for command validation."""

    @pytest.mark.fast
    def test_validation_blocks_entry_high_structural_risk(self):
        """
        TaskCommand with action_type=EVACUATE, zone_hash=2, zone_2 structural_risk=0.95
        validate_task_command returns ValidationResult with is_valid=False
        and suggested_alternative is not None.
        """
        parser = OperatorParser()

        # High structural risk zone
        world_model = {
            2: ZoneThreatAssessment(
                zone_hash=2,
                threat_score=0.95,
                human_present=True,
                human_vulnerability=0.9,
                fire_intensity=0.8,
                structural_risk=0.95,  # Very high structural risk
                rf_detection_probability=1.0,
                time_to_untenable=30.0,
                timestamp=0.0
            )
        }

        command = TaskCommand(
            command_id="test",
            zone_hash=2,
            action_type="EVACUATE",
            target_position=[0.0, 0.0, 3.0],
            priority=10,
            timeout_seconds=60.0,
            issued_by="HUMAN",
            requires_entry=True  # Attempting entry
        )

        validation = parser.validate_task_command(command, world_model)

        assert validation.is_valid is False, \
            "High structural risk should block entry"
        assert validation.suggested_alternative is not None, \
            "Should provide safe alternative action"

    @pytest.mark.fast
    def test_validation_allows_boundary_action_always(self):
        """
        TaskCommand with action_type=RELAY, requires_entry=False
        validate_task_command always returns is_valid=True
        regardless of structural_risk or fire_intensity.
        """
        parser = OperatorParser()

        # Extremely dangerous zone
        world_model = {
            5: ZoneThreatAssessment(
                zone_hash=5,
                threat_score=1.0,
                human_present=True,
                human_vulnerability=1.0,
                fire_intensity=1.0,
                structural_risk=1.0,  # Maximum risk
                rf_detection_probability=1.0,
                time_to_untenable=10.0,
                timestamp=0.0
            )
        }

        command = TaskCommand(
            command_id="test",
            zone_hash=5,
            action_type="RELAY",
            target_position=[0.0, 0.0, 3.0],
            priority=8,
            timeout_seconds=60.0,
            issued_by="HUMAN",
            requires_entry=False  # Boundary action only
        )

        validation = parser.validate_task_command(command, world_model)

        assert validation.is_valid is True, \
            "Boundary actions (RELAY) should always be allowed"


class TestAmbiguousInputs:
    """Test suite for ambiguous or unclear inputs."""

    @pytest.mark.fast
    def test_parse_ambiguous_returns_clarification(self):
        """
        Input: "help zone 2"
        Returns ClarificationRequest with a question string
        that asks what action to take.
        """
        parser = OperatorParser()

        result = parser.parse_operator_instruction("help zone 2", world_model={})

        assert isinstance(result, ClarificationRequest), \
            f"Ambiguous input should return ClarificationRequest, got {type(result)}"
        assert hasattr(result, 'question')
        assert len(result.question) > 0, "Clarification should include a question"
        assert "zone 2" in result.question.lower() or "2" in result.question, \
            "Clarification should reference the zone mentioned"


class TestFallbackMode:
    """Test suite for fallback parsing when LLM unavailable."""

    @pytest.mark.fast
    def test_llm_fallback_when_ollama_unavailable(self):
        """
        When Ollama is not running, parse_operator_instruction uses rule-based parser
        and still returns valid TaskCommand for clear inputs.
        Never raises exception.
        """
        # Force fallback mode by creating parser that will fail LLM check
        parser = OperatorParser(force_fallback=True)

        # Test clear command in fallback mode
        result = parser.parse_operator_instruction(
            "send worker to zone 4",
            world_model={}
        )

        assert isinstance(result, (TaskCommand, ClarificationRequest)), \
            "Fallback mode should still return valid result"

        if isinstance(result, TaskCommand):
            assert result.zone_hash == 4
            assert result.action_type in ["MOVE_TO", "MARK"]  # Fallback might map differently

        # Test that fallback never raises exception
        result_gibberish = parser.parse_operator_instruction(
            "complete nonsense input!!!",
            world_model={}
        )
        assert isinstance(result_gibberish, (TaskCommand, ClarificationRequest))


class TestHumanIssuer:
    """Test suite for verifying commands are marked as human-issued."""

    @pytest.mark.fast
    def test_parse_includes_issued_by_human(self):
        """
        Any TaskCommand from parse_operator_instruction has issued_by=HUMAN.
        """
        parser = OperatorParser()

        # Test various commands
        commands_to_test = [
            "send worker to zone 1",
            "suppress fire in zone 2",
            "evacuate zone 3",
            "mark zone 4",
        ]

        for cmd_text in commands_to_test:
            result = parser.parse_operator_instruction(cmd_text, world_model={})

            if isinstance(result, TaskCommand):
                assert result.issued_by == "HUMAN", \
                    f"Operator commands must have issued_by=HUMAN, got {result.issued_by}"
