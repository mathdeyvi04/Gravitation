from collections.abc import Sequence

import pygame

from src.physics.integrator import (
    AccelerationModel,
    IntegrableBody,
    Integrator,
)


class SemiImplicitEulerIntegrator(Integrator):
    """
    Integrador Euler Semi-Implícito.

    Mantido principalmente para predições e simulações que
    deliberadamente desejem este método.
    """

    def step(
        self,
        bodies: Sequence[IntegrableBody],
        fixed_delta_time: float,
        acceleration_model: AccelerationModel,
    ) -> list[pygame.Vector2]:

        count = len(bodies)

        if count == 0:
            return []

        positions = [
            pygame.Vector2(body.position)
            for body in bodies
        ]

        velocities = [
            pygame.Vector2(body.velocity)
            for body in bodies
        ]

        masses = [
            body.mass
            for body in bodies
        ]

        accelerations = acceleration_model.compute_accelerations(
            positions,
            velocities,
            masses,
        )

        dt = fixed_delta_time

        for i, body in enumerate(bodies):

            body.previous_position.update(
                body.position
            )

            body.velocity += accelerations[i] * dt
            body.position += body.velocity * dt

        return acceleration_model.compute_accelerations(
            [body.position for body in bodies],
            [body.velocity for body in bodies],
            masses,
        )