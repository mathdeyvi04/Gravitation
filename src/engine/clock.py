import pygame

class Clock:
    """@brief Gerencia a passagem de tempo, o delta variável e o passo fixo.

    @details Fluxo esperado por frame:

    1. `clock.update()` — mede o delta e acumula tempo;
    2. leia `clock.delta_time` — lógica de delta variável;
    3. `while clock.has_fixed_step(): clock.consume_fixed_step()` —
       lógica determinística.
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
        """@brief Configura o relógio.

        @details `fixed_timestep` é o passo fixo em segundos (ex.:
        `1/120`). `max_delta_time` corta deltas anormais (janela
        minimizada, travamento). `max_fixed_steps` limita quantos passos
        fixos podem rodar em um único frame.

        @param fixed_timestep Passo fixo, em segundos.
        @param max_fps Limite de frames por segundo.
        @param max_delta_time Delta máximo aceito, em segundos.
        @param max_fixed_steps Limite de passos fixos por frame.
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
        """@brief Avança um frame: mede o delta, atualiza contadores e acumula.

        @details Deve ser chamado **uma vez** no topo de cada iteração
        do loop, antes de qualquer leitura de `delta_time` ou consumo de
        passos. O delta é limitado a `max_delta_time` para evitar saltos
        após minimizar a janela.
        """
        dt = self._clock.tick(self.max_fps) / 1000.0
        # Clamp: deltas muito grandes (janela minimizada, GC, swap)
        # fariam o acumulador disparar e o próximo `while` rodaria
        # `max_fixed_steps` de uma vez — o "espiral da morte".
        if dt > self.max_delta_time:
            dt = self.max_delta_time
        self.delta_time = dt

        self.total_time += dt
        self.frame_count += 1
        self._accumulator += dt
        # Reseta a cota de passos fixos deste frame.
        self._fixed_steps_this_frame = 0

    def has_fixed_step(self) -> bool:
        """@brief Indica se há um passo fixo disponível para consumir.

        @details Retorna `False` quando o acumulador ainda não atingiu
        `fixed_timestep` ou quando `max_fixed_steps` já foi atingido no
        frame atual — o teto evita a espiral da morte.

        @return `True` se `consume_fixed_step()` deve ser chamado.
        """
        return (
            self._fixed_steps_this_frame < self.max_fixed_steps
            and self._accumulator >= self.fixed_timestep
        )

    def consume_fixed_step(self) -> None:
        """@brief Debita um `fixed_timestep` do acumulador.

        @details Chame exatamente uma vez para cada `has_fixed_step()`
        que retornou `True`.
        """
        self._accumulator -= self.fixed_timestep
        self._fixed_steps_this_frame += 1

    @property
    def interpolation_alpha(self) -> float:
        """@brief Fração do passo fixo já acumulada, em [0, 1).

        @details Use para interpolar visualmente entre o estado anterior
        e o atual quando a taxa de frames não coincide com a de passos
        fixos.

        @return Fração no intervalo `[0, 1)`; `0.0` se o passo for
            degenerado.
        """
        if self.fixed_timestep <= 0.0:
            return 0.0
        return self._accumulator / self.fixed_timestep

    @property
    def fps(self) -> float:
        """@brief FPS médio medido pelo `pygame.time.Clock` interno.

        @return Frames por segundo medidos.
        """
        return self._clock.get_fps()

    def reset(self) -> None:
        """@brief Zera contadores e acumulador, preservando as configurações.

        @details Útil ao trocar de cena ou reiniciar uma simulação sem
        recriar o objeto. Não reinicia o `pygame.time.Clock` interno.
        """
        self.delta_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0
        self._accumulator = 0.0
        self._fixed_steps_this_frame = 0