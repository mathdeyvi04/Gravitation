from pathlib import Path
from src.engine.renderer import Renderer
import pygame


class Background:
    """@brief Fundo repetível que cobre a janela e acompanha a câmera.

    @details A textura é disposta em mosaico numa superfície do tamanho
    da janela mais uma margem de um tile em cada eixo. A cada frame o
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
        background_color: tuple[int, int, int] = (255, 255, 255),
        path: str | Path = None,
        window_size: tuple[int, int] = None,
        parallax: float = 0.2,
    ) -> None:
        """@brief Carrega a textura, monta o mosaico e guarda a paralaxe.

        @details Se `path` for fornecido, a textura é carregada, o
        mosaico é pré-montado e o fundo opera em modo imagem. Caso
        contrário, cai para um preenchimento de cor sólida em
        `draw`/`clear`.

        @param background_color Cor usada no modo sem imagem.
        @param path Caminho da textura; `None` desliga o modo imagem.
        @param window_size Tamanho fixo da janela, em pixels.
        @param parallax Fator de profundidade (0.0 = fixo na tela;
            1.0 = acompanha o mundo).
        """

        # `nan` força o primeiro `update` a recalcular o offset —
        # qualquer `cx == nan` é `False`, então o early-out não dispara.
        self._last_cam_x = float("nan")
        self._last_cam_y = float("nan")

        if path:
            self.is_img = True
            tile = pygame.image.load(str(path)).convert()
            self._tile_w, self._tile_h = tile.get_size()
            self._parallax = parallax
            self._offset_x = 0.0
            self._offset_y = 0.0
            self._canvas = self._build_canvas(tile, window_size)
            return

        self.is_img = False
        self.background_color = background_color

    @staticmethod
    def _build_canvas(
        tile: pygame.Surface,
        window_size: tuple[int, int],
    ) -> pygame.Surface:
        """@brief Pré-monta o mosaico em uma superfície `(W+tw, H+th)`.

        @details A margem extra de um tile em cada eixo garante que o
        deslocamento por módulo nunca exponha uma faixa vazia na borda.

        @param tile Tile a repetir.
        @param window_size Tamanho da janela, em pixels.
        @return Superfície com o mosaico montado.
        """
        tw, th = tile.get_size()
        width, height = window_size
        canvas = pygame.Surface((width + tw, height + th)).convert()
        for y in range(0, height + th, th):
            for x in range(0, width + tw, tw):
                canvas.blit(tile, (x, y))
        return canvas

    def update(self, camera_position: pygame.Vector2) -> None:
        """@brief Recalcula o offset a partir da posição da câmera.

        @details O módulo em Python sempre devolve valor no intervalo
        `[0, tile)`, então o offset fica sempre dentro de um tile — é o
        que permite cobrir a tela com um único `blit` no `draw`. O
        early-out por posição da câmera evita recomputar quando a
        câmera não se moveu.

        @param camera_position Posição atual da câmera em world_space.
        """
        if not self.is_img:
            return

        cx = camera_position.x
        cy = camera_position.y

        # Câmera parada: offset inalterado, nada a recalcular.
        if cx == self._last_cam_x and cy == self._last_cam_y:
            return

        self._last_cam_x = cx
        self._last_cam_y = cy

        self._offset_x = (cx * self._parallax) % self._tile_w
        self._offset_y = (cy * self._parallax) % self._tile_h

    def draw(self, renderer: Renderer) -> None:
        """@brief Blita o mosaico cobrindo a tela inteira.

        @details No modo imagem, o `blit` usa o offset negativo para
        "puxar" o mosaico para trás conforme a câmera avança. No modo
        cor, apenas preenche a superfície.

        @param renderer Renderizador alvo.
        """
        if self.is_img:
            renderer.surface.blit(
                self._canvas,
                (-self._offset_x, -self._offset_y),
            )
            return

        renderer.clear(self.background_color)