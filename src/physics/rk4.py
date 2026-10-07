from collections.abc import Sequence

import pygame

from src.physics.integrator import (
    AccelerationModel,
    IntegrableBody,
    Integrator,
)


class RK4Integrator(Integrator):
    """
    Integrador Runge-Kutta de quarta ordem.

    O estado de cada corpo é:

        posição
        velocidade

    e as derivadas são:

        dposição/dt = velocidade
        dvelocidade/dt = aceleração
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

        dt = fixed_delta_time
        half_dt = dt * 0.5
        sixth_dt = dt / 6.0

        # --------------------------------------------------------------
        # Estado inicial
        # --------------------------------------------------------------

        positions_0 = [
            pygame.Vector2(body.position)
            for body in bodies
        ]

        velocities_0 = [
            pygame.Vector2(body.velocity)
            for body in bodies
        ]

        masses = [
            body.mass
            for body in bodies
        ]

        # --------------------------------------------------------------
        # K1
        # --------------------------------------------------------------

        accelerations_1 = acceleration_model.compute_accelerations(
            positions_0,
            velocities_0,
            masses,
        )

        # --------------------------------------------------------------
        # K2
        # --------------------------------------------------------------

        positions_1 = [
            positions_0[i] +
            velocities_0[i] * half_dt
            for i in range(count)
        ]

        velocities_1 = [
            velocities_0[i] +
            accelerations_1[i] * half_dt
            for i in range(count)
        ]

        accelerations_2 = acceleration_model.compute_accelerations(
            positions_1,
            velocities_1,
            masses,
        )

        # --------------------------------------------------------------
        # K3
        # --------------------------------------------------------------

        positions_2 = [
            positions_0[i] +
            velocities_1[i] * half_dt
            for i in range(count)
        ]

        velocities_2 = [
            velocities_0[i] +
            accelerations_2[i] * half_dt
            for i in range(count)
        ]

        accelerations_3 = acceleration_model.compute_accelerations(
            positions_2,
            velocities_2,
            masses,
        )

        # --------------------------------------------------------------
        # K4
        # --------------------------------------------------------------

        positions_3 = [
            positions_0[i] +
            velocities_2[i] * dt
            for i in range(count)
        ]

        velocities_3 = [
            velocities_0[i] +
            accelerations_3[i] * dt
            for i in range(count)
        ]

        accelerations_4 = acceleration_model.compute_accelerations(
            positions_3,
            velocities_3,
            masses,
        )

        # --------------------------------------------------------------
        # Estado final
        # --------------------------------------------------------------

        positions_final: list[pygame.Vector2] = []
        velocities_final: list[pygame.Vector2] = []

        for i in range(count):

            position = positions_0[i] + (
                velocities_0[i]
                + 2.0 * velocities_1[i]
                + 2.0 * velocities_2[i]
                + velocities_3[i]
            ) * sixth_dt

            velocity = velocities_0[i] + (
                accelerations_1[i]
                + 2.0 * accelerations_2[i]
                + 2.0 * accelerations_3[i]
                + accelerations_4[i]
            ) * sixth_dt

            positions_final.append(position)
            velocities_final.append(velocity)

        # --------------------------------------------------------------
        # Copia para os corpos reais
        # --------------------------------------------------------------

        for i, body in enumerate(bodies):

            body.previous_position.update(
                body.position
            )

            body.position.update(
                positions_final[i]
            )

            body.velocity.update(
                velocities_final[i]
            )

        # `accelerations_4` representa a avaliação no último estágio
        # do RK4 e é adequada para o vetor de força visualizado.
        return accelerations_4