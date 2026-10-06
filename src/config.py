from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Configs:
    """Configuração da janela e do loop principal do Pygame.

    Lida por `Application`, `Clock` e `Renderer`. Não contém nada
    relativo a uma simulação específica — trocar a cena não exige
    alterar estes valores.
    """

    # --- Janela -------------------------------------------------------

    width: int = 1280
    height: int = 720
    title: str = "Pygame Framework"

    # --- Loop ---------------------------------------------------------

    # Limite de frames por segundo.
    max_fps: int = 30

    # Frequência da atualização física.
    # 120 Hz é uma boa base para simulações.
    fixed_timestep: float = 1.0 / 120.0

    # Evita que um congelamento grande faça o jogo
    # tentar processar milhares de atualizações.
    max_delta_time: float = 0.25

    # Máximo de passos físicos por frame.
    max_fixed_steps: int = 8

    # --- Visual -------------------------------------------------------

    # Cor de fundo usada quando `background_path` é `None` (ou como
    # fallback de carregamento).
    background_color: tuple[int, int, int] = (20, 20, 30)

    # Imagem de fundo opcional. Se `None`, apenas `background_color`
    # é aplicada.
    background_path: Path | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Configuração do mundo e da física da simulação gravitacional.

    Lida pelos corpos (`MassObject`) e pelo sistema de N-corpos. Não
    contém nada relativo à janela ou ao loop — trocar a simulação
    significa trocar esta instância, não o restante da aplicação.
    """

    # --- Mundo --------------------------------------------------------

    # Dimensões do espaço em unidades de mundo (não pixels).
    # A câmera decide quanto do mundo cabe na tela.
    world_width: float = 2000.0
    world_height: float = 2000.0

    # --- Geração aleatória de corpos ---------------------------------

    # Faixas (mínimo, máximo) usadas quando `MassObject` recebe `None`.
    # `mass_range` e `radius_range` são lidas em conjunto: se só a massa
    # é informada, o raio é derivado dela por r ∝ m^(1/3).
    mass_range: tuple[float, float] = (1.0, 50.0)
    radius_range: tuple[float, float] = (4.0, 20.0)
    velocity_range: tuple[float, float] = (0.0, 30.0)

    # --- Física -------------------------------------------------------

    # Constante gravitacional em unidades arbitrárias do mundo.
    # Ajuste junto com as massas para obter a escala de tempo desejada.
    gravitational_constant: float = 2000.0

    # Suavização de Plummer aplicada no cálculo da gravidade.
    # Evita forças explodirem quando dois corpos se aproximam muito.
    # Use 0.0 para desligar (comportamento newtoniano puro).
    gravitational_softening: float = 1.0

    # Quantidade máxima de passos fixos usados na previsão
    # da trajetória futura.
    future_trajectory_steps: int = 600