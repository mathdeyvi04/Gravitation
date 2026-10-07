from collections.abc import Sequence

import pygame

from src.engine.camera import Camera2D
from src.entities.entity import Entity


class EntityPicker:
    """Descobre qual entidade foi clicada, com tolerância opcional.

    O mouse chega em screen_space; a entidade responde em world_space.
    A tolerância é medida em pixels na tela — o picker a converte para
    unidades de mundo usando o zoom atual, então o comportamento é
    consistente em qualquer nível de zoom.
    """

    __slots__ = ("tolerance",)

    def __init__(self, tolerance: float = 8.0) -> None:
        """Configura a folga de clique.

        `tolerance` é o raio (em pixels) ao redor de cada entidade que
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
    ) -> Entity | None:
        """Retorna a entidade selecionada pela posição de clique, ou None.

        Duas fases, em ordem de prioridade:

        1. **Hit direto.** Percorre as entidades de cima para baixo (a
           última desenhada é a mais "próxima" do usuário) e devolve a
           primeira que contém o ponto. Empate resolvido pela ordem
           inversa de desenho.
        2. **Tolerância.** Se nenhuma contém o ponto, escolhe a de
           menor `click_distance`, desde que dentro de `self.tolerance`
           convertido para world_space. Empate resolvido pela ordem
           inversa de desenho.

        O resultado de (2) nunca sobrepõe (1): se há hit direto, ele
        vence mesmo que outro corpo esteja numericamente mais próximo.
        """
        world_position = camera.screen_to_world(screen_position)

        tolerance_world = (
            camera.screen_to_world_size(self.tolerance)
            if self.tolerance > 0.0
            else 0.0
        )

        nearest: Entity | None = None
        nearest_distance = float("inf")

        for entity in reversed(entities):
            if not entity.active:
                continue

            if entity.hit_test(world_position, alpha):
                return entity

            if tolerance_world <= 0.0:
                continue

            distance = entity.click_distance(world_position, alpha)
            if distance <= tolerance_world and distance < nearest_distance:
                nearest_distance = distance
                nearest = entity

        return nearest