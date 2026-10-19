import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, Union

import pygame


class Renderer:
    """@brief Desenha primitivas e texto numa `pygame.Surface`.

    @details Cada método é uma operação independente sobre
    `self.surface`; não há estado implícito de câmera nem ciclo de frame
    obrigatório.
    """

    def __init__(
        self,
        surface: pygame.Surface,
        text_cache_size: int = 256,
    ) -> None:
        """@brief Liga o renderizador a `surface`.

        @param surface Superfície alvo dos desenhos.
        @param text_cache_size Limite de superfícies de texto em cache
            (LRU). Use `0` para desligar o cache de texto.
        """
        self.surface = surface
        if not pygame.font.get_init():
            pygame.font.init()
        self._font_cache: dict[tuple, pygame.font.Font] = {}
        self._text_cache_size = text_cache_size
        self._text_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()

    def _render_text_surface(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        antialias: bool,
        background: Optional[tuple[int, int, int]],
        rotation: float,
        cache_key: Optional[tuple],
    ) -> pygame.Surface:
        """@brief Renderiza `text` com `font`, usando cache LRU quando aplicável.

        @details `cache_key` só é fornecido quando a fonte é resolvida
        internamente — assim chamadas com uma `Font` explícita do
        chamador não poluem o cache com chaves que não a identificam.

        @param text Texto a renderizar.
        @param font Fonte usada.
        @param color Cor do texto.
        @param antialias Habilita suavização.
        @param background Cor de fundo opcional.
        @param rotation Rotação em graus (0 = sem rotação).
        @param cache_key Chave de cache, ou `None` para não cachear.
        @return Superfície com o texto renderizado.
        """
        if cache_key is not None and self._text_cache_size > 0:
            cache = self._text_cache
            cached = cache.get(cache_key)
            if cached is not None:
                # `move_to_end` mantém a política LRU: o hit mais recente
                # passa a ser o último a ser descartado.
                cache.move_to_end(cache_key)
                return cached
        else:
            cache = None

        surface = font.render(text, antialias, color, background)
        if rotation:
            surface = pygame.transform.rotate(surface, rotation)

        if cache is not None:
            cache[cache_key] = surface
            # `popitem(last=False)` descarta o item menos usado.
            if len(cache) > self._text_cache_size:
                cache.popitem(last=False)
        return surface

    # -- Frame --------------------------------------------------------

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """@brief Preenche toda a superfície com `color`.

        @param color Cor de preenchimento.
        """
        self.surface.fill(color)

    def present(self) -> None:
        """@brief Envia o buffer desenhado para a tela."""
        pygame.display.flip()

    # -- Texto --------------------------------------------------------

    def get_font(
        self,
        size: int,
        source: Optional[str] = None,
        *,
        bold: bool = False,
        italic: bool = False,
    ) -> pygame.font.Font:
        """@brief Retorna (e memoriza) uma `pygame.font.Font`.

        @details `source` aceita três formas: `None` (fonte padrão do
        pygame); caminho para `.ttf`/`.otf` existente (carregada do
        disco); ou qualquer outro texto (interpretado como nome de
        fonte do sistema via `SysFont`).

        @param size Tamanho em pontos.
        @param source Origem da fonte, conforme descrito.
        @param bold Negrito.
        @param italic Itálico.
        @return Instância de `pygame.font.Font` memorizada.
        """
        key = (source, size, bold, italic)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached

        if source is None:
            font = pygame.font.Font(None, size)
            font.set_bold(bold)
            font.set_italic(italic)
        elif os.path.isfile(source):
            font = pygame.font.Font(source, size)
            font.set_bold(bold)
            font.set_italic(italic)
        else:
            font = pygame.font.SysFont(source, size, bold=bold, italic=italic)

        self._font_cache[key] = font
        return font

    def measure_text(
        self,
        text: str,
        *,
        size: int = 24,
        font: Optional[pygame.font.Font] = None,
        font_name: Optional[str] = None,
        bold: bool = False,
        italic: bool = False,
    ) -> tuple[int, int]:
        """@brief Devolve `(largura, altura)` que `text` ocuparia na fonte.

        @param text Texto a medir.
        @param size Tamanho em pontos (ignorado se `font` for fornecida).
        @param font Fonte explícita; se `None`, resolvida por `size` e
            `font_name`.
        @param font_name Nome/caminho da fonte (ver `get_font`).
        @param bold Negrito.
        @param italic Itálico.
        @return Dimensões `(largura, altura)` em pixels.
        """
        if font is None:
            font = self.get_font(size, font_name, bold=bold, italic=italic)
        return font.size(text)

    def draw_text(
        self,
        text: str,
        position: Union[tuple[float, float], pygame.Vector2],
        color: tuple[int, int, int] = (255, 255, 255),
        *,
        size: int = 24,
        font: Optional[pygame.font.Font] = None,
        font_name: Optional[str] = None,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "topleft",
        antialias: bool = True,
        background: Optional[tuple[int, int, int]] = None,
        rotation: float = 0.0,
    ) -> pygame.Rect:
        """@brief Renderiza `text` e o blita em `position`.

        @details `anchor` é qualquer atributo de `pygame.Rect` (ex.:
        `"center"`, `"midleft"`, `"topright"`). Devolve o `Rect`
        efetivamente desenhado.

        @param text Texto a desenhar.
        @param position Ponto de ancoragem, em screen_space.
        @param color Cor do texto.
        @param size Tamanho em pontos.
        @param font Fonte explícita; desativa o cache de texto.
        @param font_name Nome/caminho da fonte (ver `get_font`).
        @param bold Negrito.
        @param italic Itálico.
        @param anchor Atributo de `pygame.Rect` usado como âncora.
        @param antialias Habilita suavização.
        @param background Cor de fundo opcional.
        @param rotation Rotação em graus.
        @return `Rect` ocupado pelo texto na superfície.
        """
        if font is None:
            font = self.get_font(size, font_name, bold=bold, italic=italic)
            # Cache só quando a fonte é resolvida por parâmetros
            # reproduzíveis — `Font` explícita fica de fora.
            cache_key = (
                text,
                font_name,
                size,
                bold,
                italic,
                color,
                antialias,
                background,
                rotation,
            )
        else:
            cache_key = None

        rendered = self._render_text_surface(
            text, font, color, antialias, background, rotation, cache_key,
        )
        rect = rendered.get_rect(
            **{anchor: (round(position[0]), round(position[1]))}
        )
        self.surface.blit(rendered, rect)
        return rect

    # -- Formas -------------------------------------------------------

    def draw_rect(
        self,
        color: tuple[int, int, int],
        rect: pygame.Rect,
        width: int = 0,
    ) -> None:
        """@brief Desenha um retângulo.

        @param color Cor do traço/preenchimento.
        @param rect Retângulo em screen_space.
        @param width `0` preenche; `>0` desenha apenas a borda.
        """
        pygame.draw.rect(self.surface, color, rect, width)

    def draw_circle(
        self,
        color: tuple[int, int, int],
        position: Union[tuple[float, float], pygame.Vector2],
        radius: float,
        width: int = 0,
    ) -> None:
        """@brief Desenha um círculo.

        @param color Cor do traço/preenchimento.
        @param position Centro em screen_space.
        @param radius Raio em pixels.
        @param width `0` preenche; `>0` desenha apenas a borda.
        """
        pygame.draw.circle(
            self.surface,
            color,
            (round(position[0]), round(position[1])),
            round(radius),
            width,
        )

    def draw_line(
        self,
        color: tuple[int, int, int],
        start: Union[tuple[float, float], pygame.Vector2],
        end: Union[tuple[float, float], pygame.Vector2],
        width: int = 1,
    ) -> None:
        """@brief Desenha uma linha entre `start` e `end`.

        @param color Cor da linha.
        @param start Início em screen_space.
        @param end Fim em screen_space.
        @param width Espessura em pixels.
        """
        pygame.draw.line(
            self.surface,
            color,
            (round(start[0]), round(start[1])),
            (round(end[0]), round(end[1])),
            width,
        )

    def draw_arrow(
        self,
        color: tuple[int, int, int],
        start: Union[tuple[float, float], pygame.Vector2],
        end: Union[tuple[float, float], pygame.Vector2],
        width: int = 2,
        head_length: float = 10.0,
    ) -> None:
        """@brief Desenha uma seta em screen_space.

        @param color Cor da seta.
        @param start Origem em screen_space.
        @param end Ponta em screen_space.
        @param width Espessura do traço.
        @param head_length Comprimento da ponta, em pixels.
        """
        start = pygame.Vector2(start)
        end = pygame.Vector2(end)

        direction = end - start

        if direction.length_squared() <= 1e-12:
            return

        length = direction.length()
        direction.scale_to_length(1.0)

        # Limita a ponta a metade do comprimento — em setas curtas, uma
        # ponta do tamanho nominal viraria um triângulo desproporcional.
        head_length = min(head_length, length * 0.5)

        # Os dois vértices da base da ponta: o vetor unitário rotacionado
        # ±30° a partir da ponta.
        left = end - direction.rotate(30) * head_length
        right = end - direction.rotate(-30) * head_length

        pygame.draw.line(
            self.surface,
            color,
            (round(start.x), round(start.y)),
            (round(end.x), round(end.y)),
            width,
        )

        pygame.draw.polygon(
            self.surface,
            color,
            [
                (round(end.x), round(end.y)),
                (round(left.x), round(left.y)),
                (round(right.x), round(right.y)),
            ],
        )

    def draw_polygon(
        self,
        color: tuple[int, int, int],
        points: Sequence[Union[tuple[float, float], pygame.Vector2]],
        width: int = 0,
    ) -> None:
        """@brief Desenha um polígono.

        @param color Cor do traço/preenchimento.
        @param points Vértices em screen_space, na ordem do contorno.
        @param width `0` preenche; `>0` desenha apenas o contorno.
        """
        pygame.draw.polygon(
            self.surface,
            color,
            [
                (
                    round(point[0]),
                    round(point[1]),
                )
                for point in points
            ],
            width,
        )

    def draw_polyline(
            self,
            color: tuple[int, int, int],
            points: list[tuple[float, float]],
            closed: bool = False,
            width: int = 1,
    ) -> None:
        """@brief Desenha uma sequência de segmentos conectados.

        @details Cada par consecutivo de `points` vira um segmento; se
        `closed` for `True`, um segmento adicional liga o último ao
        primeiro vértice.

        @param color Cor dos segmentos.
        @param points Vértices em screen_space.
        @param closed Fecha o contorno.
        @param width Espessura em pixels.
        """
        # `pygame.draw.lines` exige pelo menos dois vértices.
        if len(points) < 2:
            return
        pygame.draw.lines(self.surface, color, closed, points, width)