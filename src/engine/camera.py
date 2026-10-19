import pygame
from typing import Union
ScalarOrVec = Union[float, tuple[float, float], pygame.Vector2]

class Camera2D:
    """@brief Câmera 2D que transforma coordenadas entre world/screen space.

    @details A posição da câmera representa o ponto do mundo que fica no
    centro da janela. A câmera não desenha nada e não modifica os objetos
    do mundo — apenas converte coordenadas.
    """

    # Limites de zoom. O mínimo é escolhido para que, com o viewport
    # atual (1500×1000), caibam o sistema estelar inteiro e uma boa
    # margem de vazio ao redor. O máximo permite inspecionar o foguete
    # com folga.
    MIN_ZOOM: float = 0.001
    MAX_ZOOM: float = 10.0

    def __init__(
        self,
        viewport_size: tuple[int, int],
        position: pygame.Vector2 | None = None,
        zoom: float = 1.0,
    ) -> None:
        """@brief Cria a câmera com viewport, posição e zoom iniciais.

        @param viewport_size Tamanho da janela em pixels.
        @param position Ponto do mundo que fica no centro; origem se
            `None`.
        @param zoom Fator de escala (> 0).
        """
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
        """@brief Ponto central da janela em screen_space.

        @return `viewport_size * 0.5`.
        """
        return self.viewport_size * 0.5

    def world_to_screen(
        self,
        world_position: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """@brief Converte uma posição de world_space para screen_space.

        @param world_position Posição em world_space.
        @return Posição correspondente em screen_space.
        """
        world_position = pygame.Vector2(world_position)

        # Translação para o referencial da câmera, escala e
        # centralização na janela — nesta ordem.
        return (
            (world_position - self.position) * self.zoom
            + self.viewport_center
        )

    def screen_to_world(
        self,
        screen_position: pygame.Vector2 | tuple[float, float],
    ) -> pygame.Vector2:
        """@brief Converte uma posição de screen_space para world_space.

        @param screen_position Posição em screen_space.
        @return Posição correspondente em world_space.
        """
        screen_position = pygame.Vector2(screen_position)

        # Inverso exato de `world_to_screen`: recentraliza, desfaz a
        # escala e desfaz a translação.
        return (
            (screen_position - self.viewport_center) / self.zoom
            + self.position
        )

    def world_to_screen_size(
        self,
        world_size: ScalarOrVec,
    ) -> Union[float, pygame.Vector2]:
        """@brief Converte uma dimensão de world_space para screen_space.

        @details Aceita `float` (raio, comprimento) ou vetor/tupla
        (largura, altura). O tipo de retorno acompanha o da entrada.

        @param world_size Dimensão em unidades de mundo.
        @return Dimensão em pixels (mesmo tipo da entrada).
        """
        if isinstance(world_size, (int, float)):
            return world_size * self.zoom
        return pygame.Vector2(world_size) * self.zoom

    def screen_to_world_size(
        self,
        screen_size: ScalarOrVec,
    ) -> Union[float, pygame.Vector2]:
        """@brief Converte uma dimensão de screen_space para world_space.

        @details Mesma política de tipo de `world_to_screen_size`.

        @param screen_size Dimensão em pixels.
        @return Dimensão em unidades de mundo (mesmo tipo da entrada).
        """
        if isinstance(screen_size, (int, float)):
            return screen_size / self.zoom
        return pygame.Vector2(screen_size) / self.zoom

    def move(self, offset: pygame.Vector2 | tuple[float, float]) -> None:
        """@brief Move a câmera no world_space.

        @param offset Deslocamento em unidades de mundo.
        """
        self.position += pygame.Vector2(offset)

    def set_position(
        self,
        position: pygame.Vector2 | tuple[float, float],
    ) -> None:
        """@brief Define diretamente a posição da câmera no world_space.

        @param position Novo centro do viewport em world_space.
        """
        self.position.update(position)

    def set_zoom(self, zoom: float) -> None:
        """@brief Define o zoom da câmera.

        @param zoom Novo fator de escala (> 0).
        """
        if zoom <= 0.0:
            raise ValueError("zoom deve ser maior que zero.")

        self.zoom = float(zoom)

    def zoom_at(
        self,
        screen_position: pygame.Vector2 | tuple[float, float],
        factor: float,
    ) -> None:
        """@brief Multiplica o zoom por `factor`, ancorando em `screen_position`.

        @details O ponto do mundo sob `screen_position` permanece sob
        ele após o ajuste — comportamento esperado em zoom por scroll.
        `factor > 1` aproxima, `factor < 1` afasta. O novo zoom é
        limitado a `[MIN_ZOOM, MAX_ZOOM]`; se o limite já está saturado,
        nada muda (evita deslocar a câmera quando o zoom não mudaria).

        @param screen_position Âncora do zoom em screen_space.
        @param factor Fator multiplicativo do zoom (> 0).
        """
        if factor <= 0.0:
            return

        new_zoom = self.zoom * factor
        if new_zoom < self.MIN_ZOOM:
            new_zoom = self.MIN_ZOOM
        elif new_zoom > self.MAX_ZOOM:
            new_zoom = self.MAX_ZOOM

        # Sem mudança real de zoom: não desloca a câmera.
        if new_zoom == self.zoom:
            return

        # Ajuste para manter o ponto do mundo sob o cursor fixo:
        # mede-se o deslocamento introduzido pela mudança de escala e
        # compensa-se na posição da câmera.
        world_before = self.screen_to_world(screen_position)
        self.zoom = new_zoom
        world_after = self.screen_to_world(screen_position)
        self.position += world_before - world_after

    def set_viewport_size(
        self,
        viewport_size: tuple[int, int],
    ) -> None:
        """@brief Atualiza o tamanho da área visível.

        @param viewport_size Novo tamanho da janela em pixels.
        """
        self.viewport_size.update(viewport_size)

    def world_view_bounds(
            self,
    ) -> tuple[float, float, float, float]:
        """@brief Retorna os limites visíveis da câmera em world_space.

        @return `(min_x, min_y, max_x, max_y)`.
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
        """@brief Indica se um AABB em world_space intersecta a região visível.

        @details Teste de separação de eixos: dois AABBs se sobrepõem se
        nenhum dos quatro casos de afastamento acontece.

        @param bounds AABB do objeto, em `(min_x, min_y, max_x, max_y)`.
        @return `True` se o AABB intersecta a região visível.
        """
        view_min_x, view_min_y, view_max_x, view_max_y = self.world_view_bounds()

        object_min_x, object_min_y, object_max_x, object_max_y = bounds

        # Separação em qualquer eixo ⇒ sem interseção.
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
        """@brief Indica se um círculo em world_space intersecta a região visível.

        @details `radius` é a folga em unidades de mundo. Para um ponto
        puro, passe zero. Implementado como um AABB — para círculos na
        borda da tela é conservador (aceita como visível quem só encosta
        nos cantos), mas evita a raiz quadrada e mantém a checagem
        barata.

        @param world_position Centro do círculo em world_space.
        @param radius Raio em unidades de mundo.
        @return `True` se o círculo intersecta a região visível.
        """
        x, y = pygame.Vector2(world_position)

        # Círculo → AABB do mesmo tamanho; `is_world_bounds_visible`
        # faz a checagem conservadora.
        return self.is_world_bounds_visible(
            (x - radius, y - radius, x + radius, y + radius)
        )