from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import pygame


class RelativeBody(Protocol):
    position: pygame.Vector2
    velocity: pygame.Vector2
    mass: float


@dataclass(frozen=True, slots=True)
class RelativeOrbit2D:
    """
    Estado orbital relativo instantâneo entre dois corpos.

    Todos os cálculos permanecem em coordenadas cartesianas (x, y).
    """

    relative_position: pygame.Vector2
    relative_velocity: pygame.Vector2

    distance: float
    relative_speed: float

    specific_angular_momentum: float

    eccentricity: float | None
    periapsis_distance: float | None
    apoapsis_distance: float | None
    semi_major_axis: float | None
    # Período orbital kepleriano, em segundos do mundo. `None` quando
    # a órbita não é fechada (parabólica ou hiperbólica) ou quando os
    # elementos orbitais são indefinidos.
    orbital_period: float | None

    orbit_type: str

    # `None` quando a órbita já é aberta ou indefinida.
    escape_delta_v: pygame.Vector2 | None

    @classmethod
    def from_bodies(
        cls,
        body: RelativeBody,
        reference: RelativeBody,
        gravitational_constant: float,
    ) -> RelativeOrbit2D:

        relative_position = (
            body.position - reference.position
        )

        relative_velocity = (
            body.velocity - reference.velocity
        )

        distance_squared = (
            relative_position.length_squared()
        )

        distance = math.sqrt(distance_squared)
        relative_speed = relative_velocity.length()

        mu = gravitational_constant * (
            body.mass + reference.mass
        )

        # Sem distância/mu válidos não há elementos orbitais definidos.
        if distance <= 1e-12 or mu <= 0.0:
            return cls(
                relative_position=relative_position,
                relative_velocity=relative_velocity,
                distance=distance,
                relative_speed=relative_speed,
                specific_angular_momentum=0.0,
                eccentricity=None,
                periapsis_distance=None,
                apoapsis_distance=None,
                semi_major_axis=None,
                orbital_period=None,
                orbit_type="Indefinido",
                escape_delta_v=None,
            )

        r_dot_v = (
            relative_position.x * relative_velocity.x
            + relative_position.y * relative_velocity.y
        )

        velocity_squared = (
            relative_velocity.length_squared()
        )

        # Momento angular específico: h = r × v.
        angular_momentum = (
            relative_position.x * relative_velocity.y
            - relative_position.y * relative_velocity.x
        )

        # Vetor excentricidade em coordenadas cartesianas.
        eccentricity_x = (
            (velocity_squared - mu / distance)
            * relative_position.x
            - r_dot_v * relative_velocity.x
        ) / mu

        eccentricity_y = (
            (velocity_squared - mu / distance)
            * relative_position.y
            - r_dot_v * relative_velocity.y
        ) / mu

        eccentricity = math.sqrt(
            eccentricity_x * eccentricity_x
            + eccentricity_y * eccentricity_y
        )

        # p = h² / μ
        semi_latus_rectum = (
            angular_momentum * angular_momentum
        ) / mu

        periapsis_distance = (
            semi_latus_rectum
            / (1.0 + eccentricity)
        )

        # Energia específica: ε = v²/2 - μ/r
        specific_energy = (
            velocity_squared * 0.5
            - mu / distance
        )

        semi_major_axis: float | None

        if abs(specific_energy) > 1e-12:
            semi_major_axis = (
                -mu
                / (2.0 * specific_energy)
            )
        else:
            semi_major_axis = None

        # Apenas órbitas fechadas possuem apoapsis.
        if eccentricity < 1.0:
            apoapsis_distance = (
                semi_latus_rectum
                / (1.0 - eccentricity)
            )
        else:
            apoapsis_distance = None

        if eccentricity < 1e-6:
            orbit_type = "Circular"
        elif eccentricity < 1.0 - 1e-6:
            orbit_type = "Elíptica"
        elif abs(eccentricity - 1.0) <= 1e-6:
            orbit_type = "Parabólica"
        else:
            orbit_type = "Hiperbólica"

        # Escape Δv: só definido em órbitas fechadas (ligadas).
        # Direção ótima = direção da velocidade relativa atual.
        if eccentricity < 1.0 - 1e-6 and relative_speed > 1e-6:
            escape_speed = math.sqrt(2.0 * mu / distance)
            scale = (escape_speed - relative_speed) / relative_speed
            escape_delta_v = relative_velocity * scale
        else:
            escape_delta_v = None

        # Período orbital: válido apenas em órbitas ligadas, ou seja,
        # quando o semieixo maior é positivo (elipse ou círculo).
        # Parabólicas e hiperbólicas não têm período — `a` é indefinido
        # ou negativo, respectivamente.
        if semi_major_axis is not None and semi_major_axis > 0.0:
            orbital_period = 2.0 * math.pi * math.sqrt(
                semi_major_axis ** 3 / mu
            )
        else:
            orbital_period = None

        return cls(
            relative_position=relative_position,
            relative_velocity=relative_velocity,
            distance=distance,
            relative_speed=relative_speed,
            specific_angular_momentum=angular_momentum,
            eccentricity=eccentricity,
            periapsis_distance=periapsis_distance,
            apoapsis_distance=apoapsis_distance,
            semi_major_axis=semi_major_axis,
            orbit_type=orbit_type,
            escape_delta_v=escape_delta_v,
            orbital_period=orbital_period
        )