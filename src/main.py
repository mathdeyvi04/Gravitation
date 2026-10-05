from src.config import Configs
from src.engine.application import Application


class SmokeApp(Application):
    """Aplicação mínima para validar que a janela abre e o loop roda."""

    def fixed_update(self, fixed_delta_time: float) -> None:
        pass

    def __init__(self, config: Configs) -> None:
        super().__init__(config)
        self.t = 0.0

    def update(self, delta_time: float) -> None:
        self.t += delta_time

    def render(self, renderer) -> None:
        renderer.draw_text(
            f"t = {self.t:.2f}s",
            (20, 20),
            color=(240, 240, 240),
            size=28,
        )

def main() -> None:
    config = Configs(
        width=1000,
        height=600,
        title="Smoke Test",
        background_color=(20, 50, 30),
        fixed_timestep=1.0 / 120.0,
        max_fps=60,
        max_delta_time=0.25,
        max_fixed_steps=8,
    )
    app = SmokeApp(config)
    app.run()


if __name__ == "__main__":
    main()