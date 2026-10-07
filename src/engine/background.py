from pathlib import Path
from src.engine.renderer import Renderer
import pygame


class Background:
    """Fundo repetível que cobre a janela e acompanha a câmera.

    A textura é disposta em mosaico numa superfície do tamanho da
    janela mais uma margem de um tile em cada eixo. A cada frame o
    mosaico é deslocado por `-camera_pos * parallax`, tomado módulo o
    tamanho do tile. O módulo garante que o padrão sempre se repita
    continuamente — sem bordas vazias, sem saltos visíveis.
    """

    __slots__ = (
        "is_img",
        "_canvas",
        "_tile_w",
        "_tile_h",
        "_parallax",
        "_offset_x",
        "_offset_y",
        "_last_cam_x",
        "_last_cam_y",
        "background_color"
    )

    def __init__(
        self,
        background_color: tuple[int, int, int] = None,
        path: str | Path = None,
        window_size: tuple[int, int] = None,
        parallax: float = 0.2,
    ) -> None:
        """Carrega a textura, monta o mosaico e guarda a paralaxe.

        `window_size` é o tamanho fixo da janela. `parallax` é o fator
        de profundidade (0.0 = fixo na tela; 1.0 = acompanha o mundo).
        """

        self._last_cam_x = float("nan")
        self._last_cam_y = float("nan")

        if path and background_color is None:
            self.is_img = True
            tile = pygame.image.load(str(path)).convert()
            self._tile_w, self._tile_h = tile.get_size()
            self._parallax = parallax
            self._offset_x = 0.0
            self._offset_y = 0.0
            self._canvas = self._build_canvas(tile, window_size)
            return

        if background_color and path is None:
            self.is_img = False
            self.background_color = background_color
            return

        raise AttributeError

    @staticmethod
    def _build_canvas(
        tile: pygame.Surface,
        window_size: tuple[int, int],
    ) -> pygame.Surface:
        """Pré-monta o mosaico em uma superfície (W+tw, H+th)."""
        tw, th = tile.get_size()
        W, H = window_size
        canvas = pygame.Surface((W + tw, H + th)).convert()
        for y in range(0, H + th, th):
            for x in range(0, W + tw, tw):
                canvas.blit(tile, (x, y))
        return canvas

    def update(self, camera_position: pygame.Vector2) -> None:
        """Recalcula o offset a partir da posição da câmera.

        O módulo em Python sempre devolve valor no intervalo [0, tile),
        então o offset fica sempre dentro de um tile — é o que permite
        cobrir a tela com um único `blit`.
        """
        if not self.is_img:
            return

        cx = camera_position.x
        cy = camera_position.y

        if cx == self._last_cam_x and cy == self._last_cam_y:
            return

        self._last_cam_x = cx
        self._last_cam_y = cy

        self._offset_x = -(cx * self._parallax) % self._tile_w
        self._offset_y = -(cy * self._parallax) % self._tile_h

    def draw(self, renderer: Renderer) -> None:
        """Blita o mosaico cobrindo a tela inteira."""
        if self.is_img:
            renderer.surface.blit(
                self._canvas,
                (-self._offset_x, -self._offset_y),
            )
            return

        renderer.clear(self.background_color)
