from abc import ABC, abstractmethod

import pygame

from src.engine.camera import Camera2D
from src.engine.renderer import Renderer


class Entity(ABC):
    """
    Base abstrata para os objetos da aplicação.

    Subclasses devem implementar `update`, `fixed_update` e `render`.
    O ciclo de vida (`active`) é gerenciado aqui.
    """

    def __init__(
        self,
        position: pygame.Vector2 | None = None,
    ) -> None:
        """
        Inicializa posição, velocidade nula e estado ativo.
        """
        self.position = (
            pygame.Vector2(position)
            if position is not None
            else pygame.Vector2(0, 0)
        )

        self.velocity = pygame.Vector2(0, 0)
        self.active = True

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """
        Atualização lógica por frame.
        """

    @abstractmethod
    def fixed_update(self, fixed_delta_time: float) -> None:
        """
        Atualização física em timestep fixo.
        """

    @abstractmethod
    def render(
            self,
            renderer: Renderer,
            camera: Camera2D,
            alpha: float = 1.0,
    ) -> None:
        """Desenha a entidade.

        `alpha` é o fator de interpolação em [0, 1) entre o estado físico
        anterior e o atual, útil quando a taxa de frames difere da de
        passos fixos. Subclasses que não interpolam podem ignorá-lo.
        """

    def destroy(self) -> None:
        """
        Marca a entidade como inativa e dispara `on_destroy`.
        """
        if not self.active:
            return

        self.active = False
        self.on_destroy()

    def on_destroy(self) -> None:
        """
        Hook de limpeza opcional.
        """