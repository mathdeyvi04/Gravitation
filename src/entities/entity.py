from abc import ABC, abstractmethod

import pygame

from src.engine.camera import Camera2D
from src.engine.renderer import Renderer


class Entity(ABC):
    """@brief Base abstrata para os objetos da aplicação.

    @details Subclasses devem implementar `update`, `fixed_update` e
    `render`. O ciclo de vida (`active`) é gerenciado aqui.
    """

    def __init__(
        self,
        position: pygame.Vector2 | None = None,
    ) -> None:
        """@brief Inicializa posição, velocidade nula e estado ativo.

        @param position Posição inicial em world_space; origem se None.
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
        """@brief Atualização lógica por frame.

        @param delta_time Tempo desde o último frame, em segundos.
        """

    @abstractmethod
    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Atualização física em timestep fixo.

        @param fixed_delta_time Passo fixo em segundos.
        """

    @abstractmethod
    def render(
            self,
            renderer: Renderer,
            camera: Camera2D,
            alpha: float = 1.0,
    ) -> None:
        """@brief Desenha a entidade.

        @details `alpha` interpola entre o estado físico anterior e o
        atual, útil quando a taxa de frames difere da de passos fixos.
        Subclasses que não interpolam podem ignorá-lo.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação em [0, 1].
        """

    def destroy(self) -> None:
        """@brief Marca a entidade como inativa e dispara `on_destroy`.

        @details Idempotente: chamadas repetidas após a primeira não
        disparam `on_destroy` novamente.
        """
        if not self.active:
            return

        self.active = False
        self.on_destroy()

    def on_destroy(self) -> None:
        """@brief Hook de limpeza opcional, chamado por `destroy`."""

    def hit_test(
            self,
            world_position: pygame.Vector2,
            alpha: float = 1.0,
    ) -> bool:
        """@brief Indica se um ponto do mundo pertence visualmente à entidade.

        @details Entidades clicáveis devem sobrescrever este método. O
        default `False` permite entidades não selecionáveis sem que
        precisem implementar nada.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return `True` se o ponto pertence à entidade.
        """
        return False

    def get_world_bounds(
            self,
            alpha: float = 1.0,
    ) -> tuple[
             float,
             float,
             float,
             float,
         ] | None:
        """@brief Retorna o bounding box da entidade em world_space.

        @details Entidades sem forma espacial definida podem retornar
        `None`.

        @param alpha Fator de interpolação em [0, 1].
        @return `(min_x, min_y, max_x, max_y)` ou `None`.
        """
        return None

    def is_visible(
            self,
            camera: Camera2D,
            alpha: float = 1.0,
    ) -> bool:
        """@brief Indica se a entidade intersecta a região visível da câmera.

        @details Sem `get_world_bounds`, não há como recortar — a
        entidade é considerada visível por padrão.

        @param camera Câmera com o viewport atual.
        @param alpha Fator de interpolação em [0, 1].
        @return `True` se a entidade deve ser desenhada.
        """
        bounds = self.get_world_bounds(alpha)

        # Sem forma espacial conhecida, não há recorte a aplicar.
        if bounds is None:
            return True

        return camera.is_world_bounds_visible(bounds)

    def click_distance(
            self,
            world_position: pygame.Vector2,
            alpha: float = 1.0,
    ) -> float:
        """@brief Distância do ponto à superfície clicável da entidade.

        @details Retorna `0.0` se o ponto está dentro da forma, valor
        positivo se está fora, e `float("inf")` se a entidade não tem
        forma espacial bem definida — nesse caso ela só responde a
        `hit_test` direto, não à tolerância de clique.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return Distância à superfície clicável, ou `inf` sem forma.
        """
        # `inf` desabilita a fase de tolerância do `EntityPicker`.
        return float("inf")

    def collision_circles(
            self,
            alpha: float = 1.0,
    ) -> tuple[tuple[pygame.Vector2, float], ...]:
        """@brief Círculos em world_space que compõem a forma de colisão.

        @details Tupla de pares `(centro, raio)`. `alpha` interpola
        entre o estado físico anterior e o atual, coerente com o que
        `render` desenha. Entidades sem forma colidível devolvem tupla
        vazia.

        @param alpha Fator de interpolação em [0, 1].
        @return Tupla de `(centro, raio)` em world_space.
        """
        # Tupla vazia: nenhum par de círculos colide.
        return ()

    def render_selection(
            self,
            renderer: Renderer,
            camera: Camera2D,
            alpha: float = 1.0,
    ) -> None:
        """@brief Desenha um realce visual de seleção.

        @details No-op por padrão. Subclasses com forma espacial
        sobrescrevem para desenhar algo em volta (anel, caixa,
        contorno). A `Application` chama este método no `render` quando
        a entidade é a selecionada.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação em [0, 1].
        """

    def post_integrate(self) -> None:
        """@brief Hook chamado após o passo de integração, no timestep fixo.

        @details A integração (RK4) escreve diretamente em `position` e
        `velocity` dos corpos, ignorando qualquer lógica de subclasse.
        Entidades que precisam reajustar o próprio estado depois da
        integração — como um foguete acoplado, cuja posição deve
        acompanhar o hospedeiro — sobrescrevem este método. No-op por
        padrão.
        """