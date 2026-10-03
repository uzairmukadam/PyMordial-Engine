"""Kinematic Character Controller re-exports and alias module for PyMordial Engine.

Conforms to PyMordial specifications and guidelines:
`from engine.physics.character_controller import KinematicCharacterMotor, CharacterMotorConfig`
"""

from __future__ import annotations
from engine.physics.character_motor import (
    CharacterMotor,
    CharacterMotorConfig,
    CharacterMotorState,
)

# Canonical alias as specified in PyMordial architecture guidelines
KinematicCharacterMotor = CharacterMotor

__all__ = [
    "KinematicCharacterMotor",
    "CharacterMotor",
    "CharacterMotorConfig",
    "CharacterMotorState",
]
