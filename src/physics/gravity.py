from collections.abc import Sequence

import pygame


class NewtonianGravity:
    """
    Modelo de gravitação newtoniana para um sistema de N corpos.

    O modelo trabalha com posições, velocidades e massas fornecidas
    externamente. Isso permite que um integrador avalie o campo
    gravitacional em estados intermediários sem modificar os corpos reais.
    """

    __slots__ = (
        "gravitational_constant",
        "softening",
    )

    def __init__(
        self,
        gravitational_constant: float,
        softening: float = 0.0,
    ) -> None:
        self.gravitational_constant = gravitational_constant
        self.softening = softening

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        """
        Calcula a aceleração de cada corpo no estado fornecido.

        `velocities` faz parte da interface porque futuros modelos de força
        poderão depender da velocidade, embora a gravidade atualmente não
        dependa dela.
        """
        count = len(masses)

        accelerations = [
            pygame.Vector2(0.0, 0.0)
            for _ in range(count)
        ]

        eps_sq = self.softening * self.softening
        gravitational_constant = self.gravitational_constant

        for i in range(count):
            position_i = positions[i]
            mass_i = masses[i]

            for j in range(i + 1, count):
                position_j = positions[j]
                mass_j = masses[j]

                delta = position_j - position_i

                distance_squared = delta.length_squared() + eps_sq

                if distance_squared < 1e-12:
                    continue

                inv_distance = distance_squared ** -0.5
                factor = gravitational_constant * inv_distance / distance_squared

                accelerations[i] += (
                    delta * (factor * mass_j)
                )

                accelerations[j] -= (
                    delta * (factor * mass_i)
                )

        return accelerations

    def compute_accelerations_flat(
        self,
        px: list[float],
        py: list[float],
        masses: list[float],
        ax: list[float],
        ay: list[float],
    ) -> None:
        """
        Versão otimizada para a previsão de trajetória.

        Trabalha diretamente com listas de floats para evitar a criação
        de pygame.Vector2 dentro do laço quente.
        """
        count = len(masses)

        eps_sq = self.softening * self.softening
        gravitational_constant = self.gravitational_constant

        for i in range(count):
            ax[i] = 0.0
            ay[i] = 0.0

        for i in range(count):
            xi = px[i]
            yi = py[i]
            mass_i = masses[i]

            for j in range(i + 1, count):
                dx = px[j] - xi
                dy = py[j] - yi

                distance_squared = (
                    dx * dx +
                    dy * dy +
                    eps_sq
                )

                if distance_squared < 1e-12:
                    continue

                inv_distance = distance_squared ** -0.5
                factor = gravitational_constant * inv_distance / distance_squared

                force_scale_i = factor * masses[j]
                force_scale_j = factor * mass_i

                ax[i] += dx * force_scale_i
                ay[i] += dy * force_scale_i

                ax[j] -= dx * force_scale_j
                ay[j] -= dy * force_scale_j