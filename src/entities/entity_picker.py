from collections.abc import Sequence

import pygame

from src.engine.camera import Camera2D
from src.entities.entity import Entity


class EntityPicker:
    """@brief Descobre qual entidade foi clicada, com tolerância opcional.

    @details O mouse chega em screen_space; a entidade responde em
    world_space. A tolerância é medida em pixels na tela — o picker a
    converte para unidades de mundo usando o zoom atual, então o
    comportamento é consistente em qualquer nível de zoom.
    """

    __slots__ = ("tolerance",)

    def __init__(self, tolerance: float = 8.0) -> None:
        """@brief Configura a folga de clique.

        @param tolerance Raio, em pixels, ao redor de cada entidade que
            ainda conta como clique. Use `0.0` para exigir hit direto e
            desativar o mecanismo por completo.
        """
        if tolerance < 0.0:
            raise ValueError("tolerance não pode ser negativa.")
        self.tolerance = tolerance

    def pick(
        self,
        screen_position: pygame.Vector2,
        camera: Camera2D,
        entities: Sequence[Entity],
        alpha: float = 1.0,
        exclude: Entity | None = None,
    ) -> Entity | None:
        """@brief Retorna a entidade selecionada pela posição de clique.

        @details As duas fases — hit direto e tolerância — são unificadas
        num único laço: `click_distance` devolve `0.0` para hit direto e
        um valor positivo para a folga, então escolher a menor distância
        dentro da tolerância já prioriza hits diretos. Empates são
        resolvidos pela ordem inversa de desenho (a última desenhada é a
        mais "próxima" do usuário).

        @param screen_position Posição do clique em screen_space.
        @param camera Câmera usada para converter para world_space.
        @param entities Entidades candidatas, na ordem de desenho.
        @param alpha Fator de interpolação entre os estados anterior e
            atual, coerente com o que está sendo desenhado.
        @param exclude Entidade que nunca deve ser selecionada e que
            bloqueia o clique (retorna `None` em vez de "atravessar").
        @return Entidade selecionada, ou `None` se nenhuma qualifica.
        """
        world_position = camera.screen_to_world(screen_position)

        # Tolerância em pixels convertida para unidades de mundo usando
        # o zoom atual; `0.0` desliga o mecanismo de folga.
        tolerance_world = (
            camera.screen_to_world_size(self.tolerance)
            if self.tolerance > 0.0
            else 0.0
        )

        nearest: Entity | None = None
        nearest_distance = float("inf")

        # Ordem inversa de desenho: a primeira encontrada é a "de cima",
        # e empates mantêm essa entidade (não sobrescrevem `nearest`).
        for entity in reversed(entities):
            if not entity.active:
                continue

            if entity is exclude:
                # A entidade excluída também bloqueia a tolerância:
                # se o clique acertaria nela (direto ou por folga),
                # devolvemos None em vez de procurar atrás dela.
                if entity.hit_test(world_position, alpha):
                    return None

                if (
                        tolerance_world > 0.0
                        and entity.click_distance(
                    world_position,
                    alpha,
                ) <= tolerance_world
                ):
                    return None

                continue

            distance = entity.click_distance(world_position, alpha)
            # `distance == 0.0` significa hit direto; qualquer hit direto
            # vence automaticamente qualquer hit por tolerância.
            if distance <= tolerance_world and distance < nearest_distance:
                nearest_distance = distance
                nearest = entity

        return nearest