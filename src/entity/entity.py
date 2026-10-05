from abc import ABC, abstractmethod
from src.engine.renderer import Renderer
import pygame



class Entity(ABC):
    """Base abstrata para os objetos.

    Subclasses devem implementar `update`, `fixed_update` e `render`.
    O ciclo de vida (`active`) é gerenciado aqui: chame `destroy()` para
    marcar a entidade como inativa. Para limpeza extra, sobrescreva o
    hook `on_destroy` em vez de `destroy`.
    """

    def __init__(self, position: pygame.Vector2 | None = None) -> None:
        """Inicializa posição, velocidade nula e estado ativo."""
        self.position = (
            pygame.Vector2(position)
            if position is not None
            else pygame.Vector2(0, 0)
        )
        self.velocity = pygame.Vector2(0, 0)
        self.active = True

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Avança a lógica da entidade em `delta_time` segundos.

        Chamada uma vez por frame, com o tempo real decorrido. Use para
        movimento, animação e qualquer comportamento dependente de tempo
        variável.
        """

    @abstractmethod
    def fixed_update(self, fixed_delta_time: float) -> None:
        """Avança a física em passo fixo de `fixed_delta_time` segundos.

        Pode ser chamada zero ou mais vezes por frame, sempre com o mesmo
        passo. Use para integração numérica estável (Euler, RK4, Verlet)
        e para tudo que precisa de reprodutibilidade.
        """

    @abstractmethod
    def render(self, renderer: Renderer) -> None:
        """Desenha a entidade na tela através de `renderer`.

        Não deve alterar o estado da simulação — apenas apresentar o
        estado atual. Toda a lógica pertence a `update`/`fixed_update`.
        """

    def destroy(self) -> None:
        """Marca a entidade como inativa e dispara `on_destroy`.

        Idempotente: chamadas repetidas são ignoradas.
        """
        if not self.active:
            return
        self.active = False
        self.on_destroy()

    def on_destroy(self) -> None:
        """Hook de limpeza opcional. Sobrescreva se necessário.

        Chamado uma única vez por `destroy`. A implementação padrão não
        faz nada.
        """