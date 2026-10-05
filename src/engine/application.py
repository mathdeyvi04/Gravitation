from abc import ABC, abstractmethod
from src.config import Configs
from src.control.keyboard import Input
from src.engine.renderer import Renderer
from src.engine.clock import Clock
from src.engine.camera import Camera2D
import pygame

class Application(ABC):
    """Esqueleto de uma aplicação Pygame com timestep fixo.

    Responsabilidades:
    - Inicializar e finalizar o Pygame.
    - Criar a janela a partir de `Configs`.
    - Executar o loop principal: entrada, update, fixed_update, render.
    - Coordenar `Input`, `Time` e `Renderer`.

    Subclasses devem implementar `update`, `fixed_update` e `render`.
    """

    def __init__(self, config: Configs) -> None:
        """Inicializa o Pygame e todos os subsistemas da aplicação."""
        self.config = config
        self.running = False

        pygame.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption(config.title)

        self.input = Input()
        self.time = Clock(
            fixed_timestep=config.fixed_timestep,
            max_fps=config.max_fps,
            max_delta_time=config.max_delta_time,
            max_fixed_steps=config.max_fixed_steps,
        )
        self.renderer = Renderer(self.screen)
        self.camera = Camera2D(
            viewport_size=(config.width, config.height)
        )

    def run(self) -> None:
        """Executa o loop principal até `stop()` ou quit ser solicitado.

        Cada iteração: mede o tempo, lê entradas, chama `update` com delta
        variável, drena os passos fixos acumulados, limpa, chama `render` e
        apresenta. `shutdown()` é sempre chamado ao final, inclusive em
        exceção.
        """
        self.running = True
        try:
            while self.running:

                self.input.update()
                if self.input.quit_requested:
                    break

                self.update(self.time.delta_time)

                while self.time.has_fixed_step():
                    self.fixed_update(self.time.fixed_timestep)
                    self.time.consume_fixed_step()

                self.renderer.clear(self.config.background_color)
                self.render(self.renderer)
                self.renderer.present()
                self.time.update()
        finally:
            self.shutdown()

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Atualização com delta variável, uma vez por frame.

        Use para lógica que não exige passo fixo: animações, HUD,
        interpolação visual, comportamento dirigido por tempo real.
        """

    @abstractmethod
    def fixed_update(self, fixed_delta_time: float) -> None:
        """Atualização determinística em passo fixo.

        Pode rodar zero ou mais vezes por frame, sempre com o mesmo
        `fixed_delta_time`. Use para física, integração numérica
        (Euler, RK4, Verlet) e qualquer coisa que precise de
        reprodutibilidade.
        """

    @abstractmethod
    def render(self, renderer: Renderer) -> None:
        """Desenha o estado atual através de `renderer`.

        Não deve alterar o estado da simulação — apenas apresentá-lo.
        """

    def stop(self) -> None:
        """Solicita o encerramento do loop principal."""
        self.running = False

    def shutdown(self) -> None:
        """Finaliza o Pygame. Chamado automaticamente por `run()`."""
        if pygame.get_init():
            pygame.quit()