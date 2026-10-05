import pygame

class Clock:
    """Gerencia a passagem de tempo, o delta variável e o passo fixo.

    Fluxo esperado por frame:
        1. clock.update()               # mede o delta e acumula tempo
        2. leia clock.delta_time        # lógica de delta variável
        3. while clock.has_fixed_step():
               clock.consume_fixed_step()
               # lógica determinística
    """

    # Apenas para diminuir o consumo de memória
    __slots__ = (
        "fixed_timestep",
        "max_fps",
        "max_delta_time",
        "max_fixed_steps",
        "delta_time",
        "total_time",
        "frame_count",
        "_clock",
        "_accumulator",
        "_fixed_steps_this_frame",
    )

    def __init__(
        self,
        fixed_timestep: float,
        max_fps: int,
        max_delta_time: float,
        max_fixed_steps: int,
    ) -> None:
        """Configura o relógio.

        `fixed_timestep` é o passo fixo em segundos (ex.: 1/120).
        `max_fps` limita a taxa de frames. `max_delta_time` corta deltas
        anormais (janela minimizada, travamento). `max_fixed_steps`
        limita quantos passos fixos podem rodar em um único frame.
        """
        self._clock = pygame.time.Clock()
        self.fixed_timestep = fixed_timestep
        self.max_fps = max_fps
        self.max_delta_time = max_delta_time
        self.max_fixed_steps = max_fixed_steps

        self.delta_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0

        self._accumulator = 0.0
        self._fixed_steps_this_frame = 0

    def update(self) -> None:
        """Avança um frame: mede o delta, atualiza contadores e acumula.

        Deve ser chamado **uma vez** no topo de cada iteração do loop,
        antes de qualquer leitura de `delta_time` ou consumo de passos.
        O delta é limitado a `max_delta_time` para evitar saltos após
        minimizar a janela.
        """
        dt = self._clock.tick(self.max_fps) / 1000.0
        if dt > self.max_delta_time:
            dt = self.max_delta_time
        self.delta_time = dt

        self.total_time += dt
        self.frame_count += 1
        self._accumulator += dt
        self._fixed_steps_this_frame = 0

    def has_fixed_step(self) -> bool:
        """Indica se há um passo fixo disponível para consumir.

        Retorna `False` quando o acumulador ainda não atingiu
        `fixed_timestep` ou quando `max_fixed_steps` já foi atingido no
        frame atual (limita o custo e evita a espiral da morte).
        """
        return (
            self._fixed_steps_this_frame < self.max_fixed_steps
            and self._accumulator >= self.fixed_timestep
        )

    def consume_fixed_step(self) -> None:
        """Debita um `fixed_timestep` do acumulador.

        Chame exatamente uma vez para cada `has_fixed_step()` que
        retornou `True`.
        """
        self._accumulator -= self.fixed_timestep
        self._fixed_steps_this_frame += 1

    @property
    def interpolation_alpha(self) -> float:
        """Fração do passo fixo já acumulada, no intervalo [0, 1).

        Use para interpolar visualmente entre o estado anterior e o
        atual quando a taxa de frames não coincide com a de passos fixos.
        """
        if self.fixed_timestep <= 0.0:
            return 0.0
        return self._accumulator / self.fixed_timestep

    @property
    def fps(self) -> float:
        """FPS médio medido pelo `pygame.time.Clock` interno."""
        return self._clock.get_fps()

    def reset(self) -> None:
        """Zera contadores e acumulador, preservando as configurações.

        Útil ao trocar de cena ou reiniciar uma simulação sem recriar o
        objeto. Não reinicia o `pygame.time.Clock` interno.
        """
        self.delta_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0
        self._accumulator = 0.0
        self._fixed_steps_this_frame = 0