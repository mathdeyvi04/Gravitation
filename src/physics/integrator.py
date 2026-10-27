from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import pygame


class IntegrableBody(Protocol):
    """@brief Interface estrutural para corpos integráveis.

    @details Qualquer objeto com estes atributos pode ser passado a um
    `Integrator`. O integrador lê e escreve `position`, `velocity` e
    `previous_position`; `mass` é usada pelo modelo de aceleração.
    """

    position: pygame.Vector2
    velocity: pygame.Vector2
    # Guardado pelo integrador para permitir interpolação no render.
    previous_position: pygame.Vector2
    mass: float


class AccelerationModel(Protocol):
    """@brief Interface estrutural para modelos de aceleração.

    @details Modelos concretos (ex.: `NewtonianGravity`,
    `CompositeAccelerationModel`) implementam `compute_accelerations`.
    """

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        """@brief Calcula a aceleração de cada corpo no estado dado.

        @details `velocities` faz parte da interface para permitir
        modelos futuros dependentes de velocidade, embora a gravidade
        newtoniana atual não a utilize.

        @param positions Posições atuais dos corpos.
        @param velocities Velocidades atuais dos corpos.
        @param masses Massas dos corpos.
        @return Lista de acelerações, na mesma ordem dos corpos.
        """
        ...


class Integrator(ABC):
    """@brief Contrato para integradores numéricos de N corpos.

    @details Subclasses implementam `step`, que avança o estado dos
    corpos em um passo fixo e devolve as acelerações do estado final.
    """

    @abstractmethod
    def step(
        self,
        bodies: Sequence[IntegrableBody],
        fixed_delta_time: float,
        acceleration_model: AccelerationModel,
    ) -> list[pygame.Vector2]:
        """@brief Executa um passo de integração.

        @details O integrador deve atualizar `previous_position`,
        `position` e `velocity` de cada corpo. As acelerações devolvidas
        correspondem ao estado final e são usadas, por exemplo, para
        atualizar `last_force` e desenhar vetores de força.

        @param bodies Corpos a integrar.
        @param fixed_delta_time Passo fixo em segundos.
        @param acceleration_model Modelo que fornece as acelerações.
        @return Acelerações correspondentes ao estado final.
        """
        # Subclasses concretas (RK4, Euler semi-implícito) sobrescrevem.
        raise NotImplementedError