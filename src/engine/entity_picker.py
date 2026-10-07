from collections.abc import Sequence

import pygame

from src.engine.camera import Camera2D
from src.entities.entity import Entity


class EntityPicker:
    """
    Responsável exclusivamente por descobrir qual entidade foi clicada.

    O mouse chega em screen_space.
    A entidade responde ao teste em world_space.
    """

    def pick(
        self,
        screen_position: pygame.Vector2,
        camera: Camera2D,
        entities: Sequence[Entity],
        alpha: float = 1.0,
    ) -> Entity | None:

        world_position = camera.screen_to_world(
            screen_position
        )

        # A última entidade desenhada é considerada a mais próxima
        # visualmente do usuário.
        for entity in reversed(entities):

            if not entity.active:
                continue

            if entity.hit_test(
                world_position,
                alpha,
            ):
                return entity

        return None