"""Unit tests for PyMordial Engine Layered Input Contexts & Rebinding."""

from __future__ import annotations
from pathlib import Path
import pytest
from engine.input import (
    InputManager,
    InputContext,
    InputContextStack,
    ActionBinding,
    Key,
    save_bindings_to_file,
    load_bindings_from_file,
)


def test_input_context_stack_priority():
    stack = InputContextStack()

    ctx_game = InputContext("Gameplay", priority=0)
    ctx_ui = InputContext("UI", priority=100, blocks_lower_contexts=True)
    ctx_veh = InputContext("Vehicle", priority=10)

    stack.push_context(ctx_game)
    stack.push_context(ctx_ui)
    stack.push_context(ctx_veh)

    active = stack.active_contexts()
    assert [c.name for c in active] == ["UI", "Vehicle", "Gameplay"]
    assert active[0].blocks_lower_contexts is True


def test_input_context_action_consumption():
    ctx_ui = InputContext("UI", priority=100)
    ctx_ui.register_action(ActionBinding("confirm", keys=[Key.RETURN]), consume_on_trigger=True)

    assert ctx_ui.should_consume("confirm")
    assert not ctx_ui.should_consume("jump")


def test_input_bindings_serialization(tmp_path: Path):
    input_mgr = InputManager(register_defaults=True)
    json_path = tmp_path / "bindings.json"

    # Export
    assert save_bindings_to_file(input_mgr._actions, json_path)
    assert json_path.exists()

    # Load
    loaded = load_bindings_from_file(json_path)
    assert loaded is not None
    assert "jump" in loaded
    assert len(loaded["jump"]["keys"]) > 0
