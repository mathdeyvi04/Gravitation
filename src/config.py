from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Configs:
    width: int = 1280
    height: int = 720

    title: str = "Pygame Framework"

    max_fps: int = 30

    # Frequência da atualização física.
    # 120 Hz é uma boa base para simulações.
    fixed_timestep: float = 1.0 / 120.0

    background_path: Path = None

    # Evita que um congelamento grande faça o jogo
    # tentar processar milhares de atualizações.
    max_delta_time: float = 0.25

    # Máximo de passos físicos por frame.
    max_fixed_steps: int = 8

    background_color: tuple[int, int, int] = (20, 20, 30)