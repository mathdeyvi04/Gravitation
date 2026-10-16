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

    A escala é pensada para que a cena caiba em tela com zoom moderado
    e a hierarquia estelar seja visível: estrela ≫ planeta ≫ lua ≫
    meteoro ≫ foguete.
    """

    # --- Mundo --------------------------------------------------------

    # Dimensões do espaço em unidades de mundo (não pixels).
    world_width: float = 10000.0
    world_height: float = 10000.0

    # --- Geração aleatória de corpos ---------------------------------

    mass_range: tuple[float, float] = (1.0, 50.0)
    radius_range: tuple[float, float] = (4.0, 20.0)
    velocity_range: tuple[float, float] = (0.0, 30.0)

    # --- Física -------------------------------------------------------

    gravitational_constant: float = 200.0

    # Softening proporcional aos menores corpos (meteoros, raio 4–8).
    # Mantém a força finita quando duas formas pequenas se tocam, sem
    # interferir na dinâmica estelar (raios na casa das centenas).
    gravitational_softening: float = 0

    # Velocidade relativa mínima para fundir duas formas em contato.
    # Abaixo disso, a sobreposição é tratada como contato suave: é o
    # que permite o foguete repousar sobre um planeta sem ser absorvido.
    merge_min_relative_speed: float = 120

    # Velocidade relativa máxima entre o foguete e um corpo para que o
    # contato seja interpretado como pouso. Deve ser igual a
    # `merge_min_relative_speed` para que não exista faixa de
    # velocidades em que o contato é silenciosamente ignorado (abaixo
    # não funde, acima não pousa). Se um dia a política de impacto for
    # refinada, este é o valor a ajustar.
    landing_max_speed: float = 150

    future_trajectory_steps: int = 2000