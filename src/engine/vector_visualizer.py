import pygame

from src.engine.camera import Camera2D
from src.engine.renderer import Renderer


class VectorVisualizer:
    """@brief Visualizador genérico de vetores do mundo.

    @details Não armazena os vetores. Apenas os transforma para
    screen_space e solicita ao `Renderer` que desenhe as setas.
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
        """@brief Configura o estilo das setas.

        @details Começa desligado (`enabled = False`); use `toggle()`
        para exibir.

        @param color Cor das setas.
        @param width Espessura do traço, em pixels.
        @param head_length Comprimento da ponta da seta, em pixels.
        """
        self.enabled = False
        self.color = color
        self.width = width
        self.head_length = head_length

    def toggle(self) -> None:
        """@brief Alterna a visibilidade dos vetores."""
        self.enabled = not self.enabled

    def draw(
            self,
            renderer: Renderer,
            camera: Camera2D,
            origin: pygame.Vector2,
            vector: pygame.Vector2,
            reference_scale: float,
            max_screen_length: float = 120.0,
    ) -> None:
        """@brief Desenha um vetor cujo início está em `origin`.

        @details `origin` e `vector` estão em world_space. O comprimento
        em tela é ancorado em **pixels**: `reference_scale` vira
        `max_screen_length` pixels e forças maiores saturam nesse teto.
        Como o resultado independe do zoom, a seta mantém a mesma
        presença visual em qualquer nível de aproximação — e nunca
        ultrapassa `max_screen_length` pixels. `reference_scale` é o
        valor de força que o chamador considera "tamanho cheio" para
        aquele corpo; como a força depende da massa, cada corpo tem a
        sua referência.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param origin Origem do vetor em world_space.
        @param vector Vetor em world_space (unidade arbitrária).
        @param reference_scale Magnitude que corresponde a
            `max_screen_length` pixels.
        @param max_screen_length Comprimento máximo em pixels.
        """
        if not self.enabled:
            return

        length_sq = vector.length_squared()
        # Vetor nulo ou referência degenerada: não há seta a desenhar.
        if length_sq <= 1e-12 or reference_scale <= 0.0:
            return

        world_length = length_sq ** 0.5
        pixel_length = (world_length / reference_scale) * max_screen_length
        # Saturação: forças maiores que a referência ficam no teto.
        if pixel_length > max_screen_length:
            pixel_length = max_screen_length

        start = camera.world_to_screen(origin)

        # Direção em world_space = direção em screen_space (a câmera
        # só translada e escala). Multiplicamos o vetor pelo fator
        # `pixel_length / world_length` para obter o deslocamento na
        # tela, sem precisar normalizar — uma multiplicação em vez de
        # uma divisão + reconstrução do vetor unitário.
        scale = pixel_length / world_length

        renderer.draw_arrow(
            self.color,
            start,
            (start.x + vector.x * scale, start.y + vector.y * scale),
            width=self.width,
            head_length=self.head_length,
        )