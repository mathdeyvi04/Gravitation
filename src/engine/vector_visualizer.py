import pygame

from src.engine.camera import Camera2D
from src.engine.renderer import Renderer


class VectorVisualizer:
    """
    Visualizador genérico de vetores do mundo.

    Não armazena os vetores. Apenas os transforma para screen_space
    e solicita ao Renderer que desenhe as setas.
    """

    __slots__ = (
        "enabled",
        "color",
        "width",
        "head_length",
    )

    def __init__(
        self,
        color: tuple[int, int, int] = (80, 220, 255),
        width: int = 2,
        head_length: float = 10.0,
    ) -> None:
        self.enabled = False
        self.color = color
        self.width = width
        self.head_length = head_length

    def toggle(self) -> None:
        """Alterna a visibilidade dos vetores."""
        self.enabled = not self.enabled

    def draw(
        self,
        renderer: Renderer,
        camera: Camera2D,
        origin: pygame.Vector2,
        vector: pygame.Vector2,
        scale: float = 1.0,
    ) -> None:
        """
        Desenha um vetor cujo início está em `origin`.

        Tanto `origin` quanto `vector` estão em world_space.
        """

        if not self.enabled:
            return

        if vector.length_squared() <= 1e-12:
            return

        start = camera.world_to_screen(origin)

        end_world = origin + vector * scale

        end = camera.world_to_screen(end_world)

        renderer.draw_arrow(
            self.color,
            start,
            end,
            width=self.width,
            head_length=self.head_length,
        )