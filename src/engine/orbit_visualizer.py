import pygame

from src.engine.camera import Camera2D
from src.engine.renderer import Renderer


class OrbitVisualizer:
    """Desenha anéis de órbitas circulares estáveis ao redor do corpo central.

    Considera o corpo ativo de maior massa como o "Sol" e desenha N
    círculos concêntricos entre `raio_min` (logo acima da superfície
    estelar) e `max_radius` (fornecido pelo chamador — em geral uma
    fração do mundo). Cada anel representa um raio onde uma órbita
    circular fechada é possível: para um corpo de massa desprezível,
    basta ter velocidade tangencial `v = sqrt(G·M/r)`.

    O visualizador não conhece tipos concretos de entidade — apenas lê
    `position`, `radius`, `mass` e `active` do corpo escolhido.
    """

    __slots__ = ("enabled", "color", "width", "orbit_count", "min_factor")

    def __init__(
        self,
        color: tuple[int, int, int] = (230, 30, 30),
        width: int = 2,
        orbit_count: int = 6,
        min_factor: float = 1.6,
    ) -> None:
        if orbit_count < 1:
            raise ValueError("orbit_count deve ser >= 1.")
        if width < 1:
            raise ValueError("width deve ser >= 1.")
        if min_factor <= 1.0:
            raise ValueError("min_factor deve ser > 1.0 (folga acima da superfície).")

        self.enabled = False
        self.color = color
        self.width = width
        self.orbit_count = orbit_count
        self.min_factor = min_factor

    def toggle(self) -> None:
        """Alterna a visibilidade das órbitas."""
        self.enabled = not self.enabled

    def draw(
        self,
        renderer: Renderer,
        camera: Camera2D,
        bodies,
        max_radius: float,
    ) -> None:
        """Desenha os anéis ao redor do corpo mais massivo visível.

        `max_radius` é o raio externo, em world_space. Anéis cujo raio
        em tela seja menor que 1 px são ignorados para evitar poluição
        visual com zoom muito afastado.
        """
        if not self.enabled or not bodies:
            return

        central = max(
            (b for b in bodies if b.active),
            key=lambda b: b.mass,
            default=None,
        )
        if central is None:
            return

        min_radius = central.radius * self.min_factor
        if max_radius <= min_radius:
            return

        if self.orbit_count == 1:
            radii = [max_radius]
        else:
            step = (max_radius - min_radius) / (self.orbit_count - 1)
            radii = [min_radius + i * step for i in range(self.orbit_count)]

        center = camera.world_to_screen(central.position)

        for radius_world in radii:
            radius_screen = camera.world_to_screen_size(radius_world)

            if radius_screen < 1.0:
                continue

            renderer.draw_circle(
                self.color,
                center,
                radius_screen,
                width=self.width,
            )