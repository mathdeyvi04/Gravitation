import os
from collections import OrderedDict
from typing import Optional, Union

import pygame


class Renderer:
    """Desenha primitivas e texto numa `pygame.Surface`.

    Cada methods é uma operação independente sobre `self.surface`; não há
    estado implícito de câmera nem ciclo de frame obrigatório.
    """

    def __init__(
        self,
        surface: pygame.Surface,
        text_cache_size: int = 256,
    ) -> None:
        """Liga o renderizador a `surface`.

        `text_cache_size` limita quantas superfícies de texto ficam em
        cache (LRU); use 0 para desligar o cache de texto.
        """
        self.surface = surface
        if not pygame.font.get_init():
            pygame.font.init()
        self._font_cache: dict[tuple, pygame.font.Font] = {}
        self._text_cache_size = text_cache_size
        self._text_cache: OrderedDict[tuple, pygame.Surface] = OrderedDict()

    def clear(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """Preenche toda a superfície com `color`."""
        self.surface.fill(color)

    def present(self) -> None:
        """Envia o buffer desenhado para a tela."""
        pygame.display.flip()

    def get_font(
        self,
        size: int,
        source: Optional[str] = None,
        *,
        bold: bool = False,
        italic: bool = False,
    ) -> pygame.font.Font:
        """Retorna (e memoriza) uma `pygame.font.Font`.

        `source` aceita três formas:
        - `None`: fonte padrão embutida do pygame;
        - caminho para um arquivo `.ttf`/`.otf` existente: carregado do
          disco via `pygame.font.Font`;
        - qualquer outro texto: interpretado como nome de fonte do sistema
          via `pygame.font.SysFont`.
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
        """Devolve `(largura, altura)` que `text` ocuparia na fonte dada."""
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
        """Renderiza `text` e o blita em `position`.

        `anchor` é qualquer atributo de `pygame.Rect` (por exemplo
        `"center"`, `"midleft"`, `"topright"`). Devolve o `Rect` que foi
        efetivamente desenhado.
        """
        if font is None:
            font = self.get_font(size, font_name, bold=bold, italic=italic)
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

    def draw_rect(
        self,
        color: tuple[int, int, int],
        rect: pygame.Rect,
        width: int = 0,
    ) -> None:
        """Desenha um retângulo; `width=0` preenche, `width>0` desenha a borda."""
        pygame.draw.rect(self.surface, color, rect, width)

    def draw_circle(
        self,
        color: tuple[int, int, int],
        position: Union[tuple[float, float], pygame.Vector2],
        radius: float,
        width: int = 0,
    ) -> None:
        """Desenha um círculo; `width=0` preenche, `width>0` desenha a borda."""
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
        """Desenha uma linha entre `start` e `end`."""
        pygame.draw.line(
            self.surface,
            color,
            (round(start[0]), round(start[1])),
            (round(end[0]), round(end[1])),
            width,
        )

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
        """Renderiza `text` com `font`, usando cache LRU quando aplicável."""
        if cache_key is not None and self._text_cache_size > 0:
            cache = self._text_cache
            cached = cache.get(cache_key)
            if cached is not None:
                cache.move_to_end(cache_key)
                return cached
        else:
            cache = None

        surface = font.render(text, antialias, color, background)
        if rotation:
            surface = pygame.transform.rotate(surface, rotation)

        if cache is not None:
            cache[cache_key] = surface
            if len(cache) > self._text_cache_size:
                cache.popitem(last=False)
        return surface