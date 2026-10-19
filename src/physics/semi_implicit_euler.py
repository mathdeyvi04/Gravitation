from collections.abc import Sequence

import pygame

from src.physics.integrator import (
    AccelerationModel,
    IntegrableBody,
    Integrator,
)


class SemiImplicitEulerIntegrator(Integrator):
    """@brief Integrador de Euler semi-implícito (simplético).

    @details Atualiza a velocidade com a aceleração atual e só então
    usa a velocidade nova para mover a posição — a ordem que dá ao
    esquema a propriedade simplética. Mantido para predições e
    simulações que deliberadamente queiram este método.
    """

    def step(
        self,
        bodies: Sequence[IntegrableBody],
        fixed_delta_time: float,
        acceleration_model: AccelerationModel,
    ) -> list[pygame.Vector2]:
        """@brief Avança `bodies` por um passo de Euler semi-implícito.

        @details Copia o estado inicial para listas paralelas (o modelo
        de aceleração recebe sequências e não deve poder mutar os corpos
        durante a avaliação), calcula as acelerações nesse estado e
        aplica a atualização in-place, atualizando `previous_position`
        para a interpolação do render.

        @param bodies Corpos a integrar; cada um expõe `position`,
            `velocity`, `previous_position` e `mass`.
        @param fixed_delta_time Passo fixo em segundos.
        @param acceleration_model Modelo que produz a aceleração de cada
            corpo a partir de `(positions, velocities, masses)`.
        @return Acelerações avaliadas no estado *inicial* do passo — as
            mesmas que foram aplicadas às velocidades.
        """

        count = len(bodies)

        # Caso degenerado: sem corpos, não há aceleração a devolver.
        # Evita construir listas vazias e chamar o modelo à toa.
        if count == 0:
            return []

        # Snapshots do estado inicial. `compute_accelerations` recebe
        # sequências e pode, em princípio, mutá-las; copiar protege os
        # `Vector2` reais dos corpos contra escrita durante a avaliação
        # e garante que todos os corpos vejam o mesmo instante.
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

            # `previous_position` guarda o estado *antes* deste passo
            # para que o render possa interpolar entre o último e o
            # atual. Precisa ser lido antes de qualquer escrita em
            # `body.position`.
            body.previous_position.update(
                body.position
            )

            # Euler semi-implícito: primeiro atualiza a velocidade e só
            # depois usa a velocidade nova para mover a posição. A ordem
            # é o que distingue este esquema do Euler explícito (que
            # usaria a velocidade antiga no deslocamento) e o que lhe
            # confere a estabilidade simplética.
            body.velocity += accelerations[i] * dt
            body.position += body.velocity * dt

        # Diferente do RK4, aqui `accelerations` corresponde ao estado
        # no *início* do passo — é o que foi de fato usado na
        # atualização das velocidades. Chamadores que exibem força
        # (ex.: `last_force`) herdam essa semântica.
        return accelerations