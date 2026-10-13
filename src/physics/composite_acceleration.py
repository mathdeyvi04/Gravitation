from collections.abc import Sequence
from src.entities.massobject import MassObject
import pygame

from src.physics.integrator import AccelerationModel


class CompositeAccelerationModel:
    """Combina o modelo base com acelerações específicas dos corpos."""

    __slots__ = (
        "base_model",
        "bodies",
        "_thrusters"
    )

    def __init__(
        self,
        base_model: AccelerationModel,
        bodies: Sequence,
    ) -> None:
        self.base_model = base_model
        self.bodies = bodies
        default = MassObject.additional_acceleration
        self._thrusters = [
            (i, body) for i, body in enumerate(bodies)
            if type(body).additional_acceleration is not default
        ]

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        accelerations = (
            self.base_model.compute_accelerations(
                positions,
                velocities,
                masses,
            )
        )

        for i, body in self._thrusters:
            extra = body.additional_acceleration(positions[i], velocities[i])
            if extra is not None:
                accelerations[i] += extra

        return accelerations