from collections.abc import Sequence

import pygame

from src.physics.integrator import (
    AccelerationModel,
    IntegrableBody,
    Integrator,
)


class RK4Integrator(Integrator):
    """@brief Integrador Runge-Kutta de quarta ordem.

    @details Resolve, para cada corpo, o sistema

        dposição/dt   = velocidade
        dvelocidade/dt = aceleração(posição, velocidade)

    em quatro estágios (K1..K4). O estado inicial nunca é mutado
    até a etapa final: os corpos reais só recebem `position` e
    `velocity` depois que todos os estágios foram avaliados. Isso
    permite que `AccelerationModel` seja consultado com estados
    intermediários sem que o chamador precise reverter nada.
    """

    def step(
        self,
        bodies: Sequence[IntegrableBody],
        fixed_delta_time: float,
        acceleration_model: AccelerationModel,
    ) -> list[pygame.Vector2]:
        """@brief Avança um passo RK4 no conjunto de corpos.

        @details Executa os quatro estágios clássicos. Cada estágio
        consulta `acceleration_model.compute_accelerations` com um
        estado hipotético montado a partir do estado inicial — os
        `positions_*` e `velocities_*` são *cópias*, não referências
        aos corpos reais. No fim, apenas as posições/velocidades
        finais são copiadas para os `bodies`, e `previous_position`
        é atualizado para a interpolação de render.

        @param bodies Corpos a integrar; a ordem é preservada no
            retorno.
        @param fixed_delta_time Passo fixo `dt`, em segundos.
        @param acceleration_model Modelo que fornece `a(pos, vel, m)`
            no estado consultado.
        @return Lista de acelerações avaliadas no **último estágio**
            (K4), na mesma ordem de `bodies`. É a aceleração mais
            próxima do estado final e serve para `last_force`
            (vetor de força visualizado).
        """
        count = len(bodies)

        if count == 0:
            return []

        dt = fixed_delta_time
        half_dt = dt * 0.5
        sixth_dt = dt / 6.0

        # --------------------------------------------------------------
        # Estado inicial
        # --------------------------------------------------------------

        # Cópias defensivas: `AccelerationModel` pode mutar o que
        # recebe (composições, filtros), então não expomos os
        # `Vector2` dos corpos reais aos estágios intermediários.
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
        # K1 — avaliação no estado inicial
        # --------------------------------------------------------------

        accelerations_1 = acceleration_model.compute_accelerations(
            positions_0,
            velocities_0,
            masses,
        )

        # --------------------------------------------------------------
        # K2 — meio passo usando apenas K1
        # --------------------------------------------------------------

        # Aqui `velocities_0` (não K1) projeta a posição, e K1 projeta
        # a velocidade — é a definição do estágio 2.
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
        # K3 — meio passo usando K2
        # --------------------------------------------------------------

        # K3 reaproveita a posição inicial + metade de *K2* (não K1),
        # e a velocidade inicial + metade de K2. Erro comum: usar
        # `velocities_1` no lugar de `velocities_2` na projeção da
        # posição — a fórmula abaixo está correta para o estágio 3.
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
        # K4 — passo inteiro usando K3
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
        # Estado final — combinação ponderada (1, 2, 2, 1) / 6
        # --------------------------------------------------------------

        positions_final: list[pygame.Vector2] = []
        velocities_final: list[pygame.Vector2] = []

        for i in range(count):
            # Posição final combina as *velocidades* dos quatro
            # estágios, não as acelerações: dpos/dt = vel.
            position = positions_0[i] + (
                velocities_0[i]
                + 2.0 * velocities_1[i]
                + 2.0 * velocities_2[i]
                + velocities_3[i]
            ) * sixth_dt

            # Velocidade final combina as *acelerações* dos quatro
            # estágios: dvel/dt = acel.
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

        # `.update()` (e não reatribuição) preserva a identidade dos
        # `Vector2` — importante para o `Rocket` acoplado, que mantém
        # referências indiretas via `_sync_to_host` e `post_integrate`.
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