import pygame
from typing import Union
ScalarOrVec = Union[float, tuple[float, float], pygame.Vector2]

class Camera2D:
    """
    Câmera 2D responsável exclusivamente por transformar coordenadas
    entre world_space e screen_space.

    A posição da câmera representa o ponto do mundo que fica no centro
    da janela.

    A câmera não desenha nada e não modifica os objetos do mundo.
    """

    # Limites de zoom para evitar que a roda deixe a câmera inutilizável.
    MIN_ZOOM: float = 0.1
    MAX_ZOOM: float = 10.0

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
        world_size: ScalarOrVec,
    ) -> Union[float, pygame.Vector2]:
        """Converte uma dimensão de world_space para screen_space.

        Aceita `float` (raio, comprimento) ou vetor/tupla (largura,
        altura). O tipo de retorno acompanha o da entrada: `float` para
        `float`, `Vector2` para vetor ou tupla.
        """
        if isinstance(world_size, (int, float)):
            return world_size * self.zoom
        return pygame.Vector2(world_size) * self.zoom

    def screen_to_world_size(
        self,
        screen_size: ScalarOrVec,
    ) -> Union[float, pygame.Vector2]:
        """Converte uma dimensão de screen_space para world_space.

        Mesma política de tipo de `world_to_screen_size`.
        """
        if isinstance(screen_size, (int, float)):
            return screen_size / self.zoom
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

    def zoom_at(
        self,
        screen_position: pygame.Vector2 | tuple[float, float],
        factor: float,
    ) -> None:
        """Multiplica o zoom por `factor`, ancorando em `screen_position`.

        O ponto do mundo sob `screen_position` permanece sob ele após o
        ajuste — comportamento esperado em zoom por scroll. `factor > 1`
        aproxima, `factor < 1` afasta. O novo zoom é limitado a
        `[MIN_ZOOM, MAX_ZOOM]`; se o limite já está saturado, nada muda
        (evita deslocar a câmera quando o zoom não mudaria).
        """
        if factor <= 0.0:
            return

        new_zoom = self.zoom * factor
        if new_zoom < self.MIN_ZOOM:
            new_zoom = self.MIN_ZOOM
        elif new_zoom > self.MAX_ZOOM:
            new_zoom = self.MAX_ZOOM

        if new_zoom == self.zoom:
            return

        world_before = self.screen_to_world(screen_position)
        self.zoom = new_zoom
        world_after = self.screen_to_world(screen_position)
        self.position += world_before - world_after

    def set_viewport_size(
        self,
        viewport_size: tuple[int, int],
    ) -> None:
        """
        Atualiza o tamanho da área visível.
        """
        self.viewport_size.update(viewport_size)

    def world_view_bounds(
            self,
    ) -> tuple[float, float, float, float]:
        """Retorna os limites visíveis da câmera em world_space.

        Ordem: `(min_x, min_y, max_x, max_y)`.
        """
        half_width = self.viewport_size.x * 0.5 / self.zoom
        half_height = self.viewport_size.y * 0.5 / self.zoom

        return (
            self.position.x - half_width,
            self.position.y - half_height,
            self.position.x + half_width,
            self.position.y + half_height,
        )

    def is_world_bounds_visible(
            self,
            bounds: tuple[float, float, float, float],
    ) -> bool:
        """Indica se um AABB em world_space intersecta a região visível.

        `bounds` segue a ordem `(min_x, min_y, max_x, max_y)`. Teste de
        separação de eixos: dois AABBs se sobrepõem sse nenhum dos quatro
        casos de afastamento acontece.
        """
        view_min_x, view_min_y, view_max_x, view_max_y = self.world_view_bounds()

        object_min_x, object_min_y, object_max_x, object_max_y = bounds

        return not (
                object_max_x < view_min_x
                or object_min_x > view_max_x
                or object_max_y < view_min_y
                or object_min_y > view_max_y
        )

    def is_visible(
            self,
            world_position: pygame.Vector2 | tuple[float, float],
            radius: float = 0.0,
    ) -> bool:
        """Indica se um círculo em world_space intersecta a região visível.

        `radius` é a folga em unidades de mundo. Para um ponto puro, passe
        zero. Implementado como um AABB — para círculos na borda da tela
        é conservador (aceita como visível quem só encosta nos cantos),
        mas evita a raiz quadrada e mantém a checagem barata.
        """
        x, y = pygame.Vector2(world_position)

        return self.is_world_bounds_visible(
            (x - radius, y - radius, x + radius, y + radius)
        )