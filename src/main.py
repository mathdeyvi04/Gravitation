from src.config import Configs
from src.engine.application import Application
from pathlib import Path


class Gravitation(Application):

    def fixed_update(self, fixed_delta_time: float) -> None:
        pass

    def __init__(self, config: Configs) -> None:
        super().__init__(config)

    def update(self, delta_time: float) -> None:
        pass
    def render(self, renderer) -> None:
        pass

def main() -> None:
    config = Configs(
        width=1500,
        height=1000,
        title="Gravitation",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    app = Gravitation(config)
    app.run()


if __name__ == "__main__":
    main()