from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import pygame


class IntegrableBody(Protocol):
    position: pygame.Vector2
    velocity: pygame.Vector2
    previous_position: pygame.Vector2
    mass: float


class AccelerationModel(Protocol):

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        ...


class Integrator(ABC):

    @abstractmethod
    def step(
        self,
        bodies: Sequence[IntegrableBody],
        fixed_delta_time: float,
        acceleration_model: AccelerationModel,
    ) -> list[pygame.Vector2]:
        """
        Executa um passo de integração.

        Retorna as acelerações correspondentes ao estado final.
        """
        raise NotImplementedError