import pygame

class Clock:
    def __init__(
        self,
        fixed_timestep: float,
        max_fps: int,
        max_delta_time: float,
        max_fixed_steps: int,
    ):
        self.clock = pygame.time.Clock()

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
        self.delta_time = self.clock.tick(self.max_fps) / 1000.0

        # Evita um delta gigantesco após minimizar/congelar a janela.
        self.delta_time = min(self.delta_time, self.max_delta_time)

        self.total_time += self.delta_time
        self.frame_count += 1

        self._accumulator += self.delta_time
        self._fixed_steps_this_frame = 0

    def has_fixed_step(self) -> bool:
        if self._fixed_steps_this_frame >= self.max_fixed_steps:
            return False

        return self._accumulator >= self.fixed_timestep

    def consume_fixed_step(self) -> None:
        self._accumulator -= self.fixed_timestep
        self._fixed_steps_this_frame += 1

    @property
    def interpolation_alpha(self) -> float:
        if self.fixed_timestep <= 0.0:
            return 0.0

        return self._accumulator / self.fixed_timestep