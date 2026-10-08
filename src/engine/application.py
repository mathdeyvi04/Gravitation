from abc import ABC, abstractmethod
from src.config import Configs
from src.control.keyboard import Input
from src.engine.background import Background
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

        # Inicializamos somente os subsistemas que a aplicação realmente
        # usa. `pygame.init()` também sobe o mixer de áudio, que jamais é
        # utilizado — e `pygame.mixer.quit()` é conhecido por travar o
        # encerramento em várias combinações de driver/SO. Evitar o mixer
        # deixa o shutdown previsível.
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption(config.title)

        self.input = Input()
        self.clock = Clock(
            fixed_timestep=config.fixed_timestep,
            max_fps=config.max_fps,
            max_delta_time=config.max_delta_time,
            max_fixed_steps=config.max_fixed_steps,
        )
        self.renderer = Renderer(self.screen)
        self.camera = Camera2D(
            viewport_size=(config.width, config.height)
        )
        self.background = Background(
            path=config.background_path,
            window_size=(config.width, config.height),
            parallax=0.2
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
                self.clock.update()

                self.input.update()
                if self.input.quit_requested:
                    break

                self.update(self.clock.delta_time)

                while self.clock.has_fixed_step():
                    self.fixed_update(self.clock.fixed_timestep)
                    self.clock.consume_fixed_step()

                self.background.update(self.camera.position)
                self.background.draw(self.renderer)
                self.render(self.renderer)
                self.renderer.present()
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
        """Finaliza os subsistemas que a aplicação inicializou.

        Fecha na ordem inversa de `__init__` para garantir que nenhuma
        superfície/fonte fique pendurada enquanto o display ainda existe.
        Não chamamos `pygame.quit()` porque ele desmontaria também o
        mixer — que nunca inicializamos — e esse é o ponto onde o
        encerramento costuma travar em certos drivers.
        """
        if pygame.font.get_init():
            pygame.font.quit()
        if pygame.display.get_init():
            pygame.display.quit()