from abc import ABC, abstractmethod
from src.config import Configs
from src.control.keyboard import Input
from src.engine.background import Background
from src.engine.renderer import Renderer
from src.engine.clock import Clock
from src.engine.camera import Camera2D
import pygame

class Application(ABC):
    """@brief Esqueleto de uma aplicação Pygame com timestep fixo.

    @details Responsabilidades:

    - Inicializar e finalizar os subsistemas que usa.
    - Criar a janela a partir de `Configs`.
    - Executar o loop principal: entrada, update, fixed_update, render.
    - Coordenar `Input`, `Clock`, `Renderer`, `Camera2D` e `Background`.

    Subclasses devem implementar `update`, `fixed_update` e `render`.
    """

    def __init__(self, config: Configs) -> None:
        """@brief Inicializa display, font e os subsistemas da aplicação.

        @details Não chamamos `pygame.init()` porque ele sobe também o
        mixer de áudio, que não usamos — e cujo shutdown trava em várias
        combinações de driver. Ver `shutdown()` para a contrapartida.

        @param config Configuração da janela e do loop.
        """
        self.config = config
        self.running = False

        # Inicialização seletiva: só os subsistemas efetivamente usados.
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
            parallax=0.05
        )

        self.speed_multiplier = 1.0

    def run(self) -> None:
        """@brief Executa o loop principal até `stop()` ou quit ser solicitado.

        @details Cada iteração: mede o tempo, lê entradas, chama `update`
        com delta variável, drena os passos fixos acumulados, desenha o
        fundo (que pode limpar a tela), chama `render` e apresenta. O
        `try/finally` garante que `shutdown()` rode mesmo se uma
        exceção escapar de uma subclasse.
        """
        self.running = True
        try:
            while self.running:
                self.clock.update()

                self.input.update()
                if self.input.quit_requested:
                    break

                self.update(self.clock.delta_time)

                # Passos fixos: `has_fixed_step` respeita `max_fixed_steps`,
                # evitando a espiral da morte após um stall.
                steps_to_run = 0
                while self.clock.has_fixed_step():
                    steps_to_run += 1
                    self.clock.consume_fixed_step()
                accelerated_steps = int(steps_to_run * self.speed_multiplier)
                if accelerated_steps == 0 and steps_to_run > 0 and self.speed_multiplier >= 1.0:
                    accelerated_steps = steps_to_run

                for _ in range(accelerated_steps):
                    self.fixed_update(self.clock.fixed_timestep)

                self.background.update(self.camera.position)
                self.background.draw(self.renderer)
                self.render(self.renderer)
                self.renderer.present()
        finally:
            self.shutdown()

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """@brief Atualização com delta variável, uma vez por frame.

        @details Use para lógica que não exige passo fixo: animações,
        HUD, interpolação visual, comportamento dirigido por tempo real.

        @param delta_time Tempo desde o último frame, em segundos.
        """

    @abstractmethod
    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Atualização determinística em passo fixo.

        @details Pode rodar zero ou mais vezes por frame, sempre com o
        mesmo `fixed_delta_time`. Use para física, integração numérica
        (Euler, RK4, Verlet) e qualquer coisa que precise de
        reprodutibilidade.

        @param fixed_delta_time Passo fixo em segundos.
        """

    @abstractmethod
    def render(self, renderer: Renderer) -> None:
        """@brief Desenha o estado atual através de `renderer`.

        @details Não deve alterar o estado da simulação — apenas
        apresentá-lo.

        @param renderer Renderizador alvo.
        """

    def stop(self) -> None:
        """@brief Solicita o encerramento do loop principal."""
        self.running = False

    def shutdown(self) -> None:
        """@brief Finaliza os subsistemas que a aplicação inicializou.

        @details Fecha na ordem inversa de `__init__` para garantir que
        nenhuma superfície/fonte fique pendurada enquanto o display
        ainda existe. Não chamamos `pygame.quit()` porque ele
        desmontaria também o mixer — que nunca inicializamos — e esse é
        o ponto onde o encerramento costuma travar em certos drivers.
        """
        if pygame.font.get_init():
            pygame.font.quit()
        if pygame.display.get_init():
            pygame.display.quit()