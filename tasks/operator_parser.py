"""
operator_parser.py
------------------
Natural language parser for operator instructions.

Converts operator text into TaskCommands using LLM (Ollama) or rule-based fallback.
"""

import re
import json
import uuid
from dataclasses import dataclass
from typing import Dict, Union, Any, Optional

from tasks.contracts import TaskCommand, ZoneThreatAssessment, ValidationResult


@dataclass
class ClarificationRequest:
    """
    Request for clarification when operator instruction is ambiguous.
    """
    question: str
    context: Optional[Dict[str, Any]] = None


class OperatorParser:
    """
    Natural language parser for operator instructions.

    Uses Ollama LLM (llama3.2:1b) for parsing with fallback to rule-based parser.
    """

    def __init__(self, force_fallback: bool = False):
        """
        Initialize operator parser.

        Args:
            force_fallback: If True, always use rule-based parser (for testing)
        """
        self.force_fallback = force_fallback
        self.ollama_available = self._check_ollama() if not force_fallback else False

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            return response.status_code == 200
        except:
            return False

    def parse_operator_instruction(
        self,
        instruction: str,
        world_model: Dict[int, ZoneThreatAssessment]
    ) -> Union[TaskCommand, ClarificationRequest]:
        """
        Parse natural language instruction into TaskCommand or ClarificationRequest.

        Never raises exceptions - always returns a valid result.

        Args:
            instruction: Natural language instruction from operator
            world_model: Current world model for context

        Returns:
            TaskCommand if instruction is clear, ClarificationRequest if ambiguous
        """
        # Handle empty or nonsense input
        if not instruction or len(instruction.strip()) < 2:
            return ClarificationRequest(
                question="Could you please provide a command? (e.g., 'send worker to zone 2')",
                context={}
            )

        try:
            if self.ollama_available:
                return self._parse_with_llm(instruction, world_model)
            else:
                return self._parse_with_rules(instruction, world_model)
        except Exception as e:
            # Fallback to rule-based if LLM fails
            try:
                return self._parse_with_rules(instruction, world_model)
            except:
                # Last resort: return clarification
                return ClarificationRequest(
                    question="I didn't understand that command. Could you rephrase? (e.g., 'suppress fire in zone 3')",
                    context={"error": str(e)}
                )

    def _parse_with_llm(
        self,
        instruction: str,
        world_model: Dict[int, ZoneThreatAssessment]
    ) -> Union[TaskCommand, ClarificationRequest]:
        """Parse instruction using Ollama LLM."""
        import requests

        # Build context from world model
        context_str = self._build_world_context(world_model)

        prompt = f"""You are a command parser for a USAR firefighting swarm system.
Parse the operator instruction into a JSON command.

Current situation:
{context_str}

Operator instruction: {instruction}

Output ONLY valid JSON with these exact fields:
{{
    "action_type": "MOVE_TO|SUPPRESS|EVACUATE|MARK|RELAY|HOLD",
    "zone_hash": <integer zone number>,
    "priority": <1-10, where 10 is highest>,
    "requires_entry": <true|false>,
    "is_ambiguous": <true|false>,
    "clarification_question": "<question if ambiguous, else null>"
}}

Rules:
- "send", "move", "go" → MOVE_TO
- "suppress", "fight", "extinguish" → SUPPRESS
- "evacuate", "rescue" → EVACUATE
- "mark", "monitor" → MARK
- "relay", "observe" → RELAY
- "hold", "wait", "pause" → HOLD
- "urgent", "immediately", "child" → priority 9-10
- "when possible" → priority 3-5
- Entry required for EVACUATE and SUPPRESS
- If unclear, set is_ambiguous=true

JSON output:"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                output_text = result.get("response", "{}")

                # Parse JSON response
                parsed = json.loads(output_text)

                if parsed.get("is_ambiguous", False):
                    return ClarificationRequest(
                        question=parsed.get("clarification_question", "Could you clarify the command?"),
                        context=parsed
                    )

                return TaskCommand(
                    command_id=str(uuid.uuid4()),
                    zone_hash=int(parsed.get("zone_hash", 0)),
                    action_type=parsed.get("action_type", "MARK"),
                    target_position=[0.0, 0.0, 3.0],
                    priority=int(parsed.get("priority", 5)),
                    timeout_seconds=120.0,
                    issued_by="HUMAN",
                    requires_entry=bool(parsed.get("requires_entry", False))
                )
        except:
            pass

        # Fallback to rules
        return self._parse_with_rules(instruction, world_model)

    def _parse_with_rules(
        self,
        instruction: str,
        world_model: Dict[int, ZoneThreatAssessment]
    ) -> Union[TaskCommand, ClarificationRequest]:
        """Parse instruction using rule-based keyword matching."""
        instruction_lower = instruction.lower()

        # Extract zone number
        zone_hash = self._extract_zone_number(instruction_lower)

        # Extract action type
        action_type = self._extract_action_type(instruction_lower)

        # Extract priority
        priority = self._extract_priority(instruction_lower)

        # Determine if entry required
        requires_entry = action_type in {"EVACUATE", "SUPPRESS"}

        # Check for ambiguity
        if zone_hash is None or action_type is None:
            # Ambiguous - ask for clarification
            if zone_hash is None:
                question = "Which zone should I target? (e.g., 'zone 2')"
            else:
                question = f"What action should I take in zone {zone_hash}? (suppress/evacuate/mark/relay)"

            return ClarificationRequest(question=question, context={})

        return TaskCommand(
            command_id=str(uuid.uuid4()),
            zone_hash=zone_hash,
            action_type=action_type,
            target_position=[0.0, 0.0, 3.0],
            priority=priority,
            timeout_seconds=120.0,
            issued_by="HUMAN",
            requires_entry=requires_entry
        )

    def _build_world_context(self, world_model: Dict[int, ZoneThreatAssessment]) -> str:
        """Build context string from world model."""
        if not world_model:
            return "No zone data available."

        lines = []
        for zone_hash, threat in sorted(world_model.items()):
            lines.append(
                f"Zone {zone_hash}: fire={threat.fire_intensity:.1f}, "
                f"human={'yes' if threat.human_present else 'no'}, "
                f"threat={threat.threat_score:.1f}"
            )
        return "\n".join(lines)

    def _extract_zone_number(self, text: str) -> Optional[int]:
        """Extract zone number from text."""
        # Try digit extraction
        match = re.search(r'zone\s+(\d+)', text)
        if match:
            return int(match.group(1))

        # Try written numbers
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        }
        for word, num in number_words.items():
            if f'zone {word}' in text:
                return num

        # Try standalone digits
        match = re.search(r'\b(\d+)\b', text)
        if match:
            return int(match.group(1))

        return None

    def _extract_action_type(self, text: str) -> Optional[str]:
        """Extract action type from text."""
        # Action keywords mapping
        if any(word in text for word in ['suppress', 'fight', 'extinguish', 'put out']):
            return "SUPPRESS"
        elif any(word in text for word in ['evacuate', 'rescue', 'save', 'get out']):
            return "EVACUATE"
        elif any(word in text for word in ['send', 'move', 'go', 'deploy']):
            return "MOVE_TO"
        elif any(word in text for word in ['mark', 'monitor', 'watch']):
            return "MARK"
        elif any(word in text for word in ['relay', 'observe', 'report']):
            return "RELAY"
        elif any(word in text for word in ['hold', 'wait', 'pause', 'stop']):
            return "HOLD"
        elif 'help' in text:
            return None  # Ambiguous
        else:
            return None

    def _extract_priority(self, text: str) -> int:
        """Extract priority from text."""
        if any(word in text for word in ['urgent', 'immediately', 'emergency', 'child', 'critical']):
            return 9
        elif any(word in text for word in ['soon', 'quickly', 'fast']):
            return 7
        elif any(word in text for word in ['when possible', 'eventually', 'later']):
            return 3
        else:
            return 5  # Default priority

    def validate_task_command(
        self,
        command: TaskCommand,
        world_model: Dict[int, ZoneThreatAssessment]
    ) -> ValidationResult:
        """
        Validate task command against current world state.

        Args:
            command: Task command to validate
            world_model: Current world model

        Returns:
            ValidationResult with validity and suggested alternatives
        """
        # Get threat assessment for target zone
        threat = world_model.get(command.zone_hash)

        if threat is None:
            # Zone not in world model - accept command
            return ValidationResult(
                is_valid=True,
                reason="Zone not yet assessed - command accepted",
                suggested_alternative=None
            )

        # Rule 1: Boundary actions (RELAY, MARK) are always allowed
        if command.action_type in {"RELAY", "MARK"} and not command.requires_entry:
            return ValidationResult(
                is_valid=True,
                reason="Boundary action - always safe",
                suggested_alternative=None
            )

        # Rule 2: High structural risk blocks entry
        if command.requires_entry and threat.structural_risk > 0.9:
            # Suggest RELAY as safe alternative
            alt_cmd = TaskCommand(
                command_id=str(uuid.uuid4()),
                zone_hash=command.zone_hash,
                action_type="RELAY",
                target_position=command.target_position,
                priority=command.priority,
                timeout_seconds=60.0,
                issued_by=command.issued_by,
                requires_entry=False
            )

            return ValidationResult(
                is_valid=False,
                reason=f"Structural risk too high ({threat.structural_risk:.2f}) for zone entry",
                suggested_alternative=alt_cmd
            )

        # Rule 3: Very high fire intensity with entry requires explicit confirmation
        if command.requires_entry and threat.fire_intensity > 0.95:
            return ValidationResult(
                is_valid=False,
                reason=f"Fire intensity critical ({threat.fire_intensity:.2f}) - confirm entry authorization",
                suggested_alternative=None
            )

        # Command is valid
        return ValidationResult(
            is_valid=True,
            reason="Command validated successfully",
            suggested_alternative=None
        )
