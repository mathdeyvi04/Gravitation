import pygame


class Camera2D:
    """
    Câmera 2D responsável exclusivamente por transformar coordenadas
    entre world_space e screen_space.

    A posição da câmera representa o ponto do mundo que fica no centro
    da janela.

    A câmera não desenha nada e não modifica os objetos do mundo.
    """

    def __init__(
        self,
        viewport_size: tuple[int, int],
        position: pygame.Vector2 | None = None,
        zoom: float = 1.0,
    ) -> None:
        if zoom <= 0.0:
            raise ValueError("zoom deve ser maior que zero.")

        self.viewport_size = pygame.Vector2(viewport_size)

        self.position = (
            pygame.Vector2(position)
            if position is not None
            else pygame.Vector2(0, 0)
        )

        self.zoom = float(zoom)

    @property
    def viewport_center(self) -> pygame.Vector2:
        """
        Ponto central da janela em screen_space.
        """
        return self.viewport_size * 0.5

    def world_to_screen(
        self,
        world_position: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """
        Converte uma posição de world_space para screen_space.
        """
        world_position = pygame.Vector2(world_position)

        return (
            (world_position - self.position) * self.zoom
            + self.viewport_center
        )

    def screen_to_world(
        self,
        screen_position: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """
        Converte uma posição de screen_space para world_space.
        """
        screen_position = pygame.Vector2(screen_position)

        return (
            (screen_position - self.viewport_center) / self.zoom
            + self.position
        )

    def world_to_screen_size(
        self,
        world_size: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """
        Converte uma dimensão de world_space para screen_space.

        Exemplo:
            tamanho de 100 unidades no mundo
            zoom = 2
            resultado = 200 pixels
        """
        return pygame.Vector2(world_size) * self.zoom

    def screen_to_world_size(
        self,
        screen_size: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """
        Converte uma dimensão de screen_space para world_space.
        """
        return pygame.Vector2(screen_size) / self.zoom

    def world_to_screen_rect(
        self,
        world_rect: pygame.Rect,
    ) -> pygame.Rect:
        """
        Converte um pygame.Rect de world_space para screen_space.
        """
        screen_position = self.world_to_screen(world_rect.topleft)
        screen_size = self.world_to_screen_size(world_rect.size)

        return pygame.Rect(
            round(screen_position.x),
            round(screen_position.y),
            round(screen_size.x),
            round(screen_size.y),
        )

    def move(self, offset: pygame.Vector2 | tuple[float, float]) -> None:
        """
        Move a câmera no world_space.
        """
        self.position += pygame.Vector2(offset)

    def set_position(
        self,
        position: pygame.Vector2 | tuple[float, float],
    ) -> None:
        """
        Define diretamente a posição da câmera no world_space.
        """
        self.position.update(position)

    def set_zoom(self, zoom: float) -> None:
        """
        Define o zoom da câmera.
        """
        if zoom <= 0.0:
            raise ValueError("zoom deve ser maior que zero.")

        self.zoom = float(zoom)

    def set_viewport_size(
        self,
        viewport_size: tuple[int, int],
    ) -> None:
        """
        Atualiza o tamanho da área visível.
        """
        self.viewport_size.update(viewport_size)