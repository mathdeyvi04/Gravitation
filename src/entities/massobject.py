import math
import pygame
import random
from typing import Optional

from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity


class MassObject(Entity):
    """Corpo com massa e raio que interage gravitacionalmente.

    `MassObject` representa o estado físico de um corpo — posição,
    velocidade, massa, raio e cor — mas **não integra a si mesmo**.
    A integração é responsabilidade de um `Integrator` externo, que lê
    e escreve esses campos em passos fixos. Essa separação permite
    trocar o método numérico (Euler, RK4, Verlet) sem tocar no corpo.

    Campos auxiliares:

    - `previous_position` guarda o estado imediatamente anterior ao
      último passo fixo, permitindo interpolação suave no render.
    - `last_force` guarda a força da última avaliação do integrador,
      usada apenas para visualização (o `VectorVisualizer` desenha a
      seta correspondente).
    """

    __slots__ = (
        "mass",
        "radius",
        "color",
        "last_force",
        "previous_position",
    )

    # Folga em pixels entre a borda do disco e o anel de seleção.
    # Constante de classe porque não muda por instância.
    SELECTION_MARGIN = 6

    def __init__(
        self,
        config: SimulationConfig,
        mass: Optional[float] = None,
        radius: Optional[float] = None,
        position: Optional[pygame.Vector2] = None,
        velocity: Optional[pygame.Vector2] = None,
        color: tuple[int, int, int] = (220, 220, 220),
    ) -> None:
        """Cria um corpo a partir de `config`.

        Qualquer parâmetro deixado como `None` é sorteado dentro das
        faixas definidas por `config`. Os sorteios são:

        - **posição**: uniforme dentro de `[0, world_width] × [0, world_height]`.
        - **massa**: uniforme em `config.mass_range`.
        - **raio**: derivado da massa por `r ∝ m^(1/3)` quando só a massa
          é informada, preservando a hipótese de densidade uniforme;
          caso `radius_range` seja degenerado (mínimo igual ao máximo),
          usa o valor mínimo.
        - **velocidade**: direção uniforme em `[0, 2π)` e módulo uniforme
          em `config.velocity_range` (distribuição isotrópica, ao
          contrário de sortear `vx` e `vy` independentes).

        `config` é obrigatório: os limites de sorteio vivem lá, não em
        defaults escondidos na classe.
        """
        world_w = config.world_width
        world_h = config.world_height
        mass_range = config.mass_range
        radius_range = config.radius_range
        speed_range = config.velocity_range

        if position is None:
            position = pygame.Vector2(
                random.uniform(0.0, world_w),
                random.uniform(0.0, world_h),
            )

        if mass is None:
            mass = random.uniform(*mass_range)

        if radius is None:
            # r ∝ m^(1/3): preserva a hipótese de densidade uniforme.
            # Sem essa derivação, corpos de massas diferentes teriam
            # densidades discrepantes e a fusão pareceria arbitrária.
            radius = radius_range[0] + (
                (radius_range[1] - radius_range[0])
                * ((mass - mass_range[0]) / (mass_range[1] - mass_range[0]))
            ) ** (1.0 / 3.0) if mass_range[1] > mass_range[0] else radius_range[0]

        if velocity is None:
            speed = random.uniform(*speed_range)
            angle = random.uniform(0.0, 2.0 * math.pi)
            velocity = pygame.Vector2(
                speed * math.cos(angle),
                speed * math.sin(angle),
            )

        super().__init__(position)
        if velocity is not None:
            self.velocity = pygame.Vector2(velocity)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.previous_position = pygame.Vector2(self.position)
        self.last_force = pygame.Vector2(0.0, 0.0)

    def additional_acceleration(
        self,
        position: pygame.Vector2,
        velocity: pygame.Vector2,
    ) -> Optional[pygame.Vector2]:
        """Aceleração adicional específica do corpo."""
        return None

    def prediction_acceleration(
        self,
        elapsed_time: float,
    ) -> Optional[pygame.Vector2]:
        """Aceleração adicional usada pela previsão futura."""
        return None

    def update(self, delta_time: float) -> None:
        """No-op: `MassObject` só reage a passos fixos.

        Toda a dinâmica do corpo é determinística e roda em
        `fixed_update` via o integrador externo. Manter este método
        vazio é o que garante que o `update` de delta variável — usado
        pela `Application` para lógica de frame — não interfira na
        física.
        """

    def fixed_update(self, fixed_delta_time: float) -> None:
        """No-op: a integração é feita externamente pelo `Integrator`.

        O `MassObject` expõe `position`, `velocity`, `previous_position`
        e `mass`, e o integrador escreve nesses campos. A escolha do
        método numérico (Euler, RK4, Verlet) pertence ao integrador,
        não ao corpo.
        """

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        """Desenha o corpo como um disco preenchido.

        A posição de desenho é interpolada entre `previous_position` e
        `position` por `alpha`, o que suaviza o movimento quando a taxa
        de render não coincide com a de passos fixos. Corpos fora da
        área visível da câmera são ignorados.
        """
        pos = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        if not self.is_visible(camera, alpha):
            return

        renderer.draw_circle(
            self.color,
            camera.world_to_screen(pos),
            camera.world_to_screen_size(self.radius),
        )

    def overlaps(self, other: "MassObject") -> bool:
        """Indica se os discos de `self` e `other` se sobrepõem.

        Compara os quadrados das distâncias com o quadrado da soma dos
        raios — evita a raiz quadrada. Colisão entre discos é
        equivalente a colisão entre os círculos que representam.
        """
        r = self.radius + other.radius
        return self.position.distance_squared_to(other.position) <= r * r

    def merge_with(self, other: "MassObject") -> None:
        """Absorve `other`, conservando massa, momento linear e volume.

        O novo centro de massa é a média ponderada das posições, e a
        nova velocidade é a média ponderada das velocidades — o que
        preserva o momento linear total do par. O raio é recalculado
        assumindo densidade uniforme (`r ∝ m^(1/3)`).

        `other` é destruído ao final. Idempotente no sentido de que
        `destroy()` já é idempotente: chamar `merge_with` sobre um
        corpo já inativo apenas o destrói e retorna sem efeito.
        """
        total = self.mass + other.mass
        if total <= 0.0:
            other.destroy()
            return

        self.position = (
            self.position * self.mass + other.position * other.mass
        ) / total
        self.velocity = (
            self.velocity * self.mass + other.velocity * other.mass
        ) / total
        self.radius = (self.radius ** 3 + other.radius ** 3) ** (1.0 / 3.0)
        self.mass = total

        # A posição mudou; sem isso, o render interpolaria da posição
        # antiga até a nova e o corpo pareceria deslizar após a fusão.
        self.previous_position.update(self.position)

        other.destroy()

    def get_world_bounds(
        self,
        alpha: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """Retorna o AABB do disco em world_space na posição interpolada.

        Ordem: `(min_x, min_y, max_x, max_y)`. Usado por `is_visible`
        para decidir se o corpo aparece na tela, e reaproveitável para
        qualquer checagem baseada em caixa.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        return (
            position.x - self.radius,
            position.y - self.radius,
            position.x + self.radius,
            position.y + self.radius,
        )

    def hit_test(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> bool:
        """Indica se o ponto em world_space está dentro do disco.

        Considera a posição interpolada por `alpha`, coerente com o que
        está sendo desenhado no mesmo frame.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        return (
            position.distance_squared_to(world_position)
            <= self.radius * self.radius
        )

    def click_distance(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> float:
        """Distância do ponto à superfície do disco (0.0 se dentro).

        Coerente com `hit_test` e `render`: usa a posição interpolada.
        É o que o `EntityPicker` consulta na fase de tolerância, quando
        nenhuma entidade contém o ponto do clique diretamente.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )
        return max(0.0, position.distance_to(world_position) - self.radius)

    def render_selection(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Desenha um anel em volta do corpo para indicar seleção.

        O anel fica a `SELECTION_MARGIN` pixels da borda do disco, uma
        folga constante em screen_space — assim o destaque permanece
        perceptível mesmo em zooms extremos, onde o disco é minúsculo.

        `color` permite usar cores diferentes para seleções distintas:
        o `main.py` usa branco para a seleção principal e laranja para
        o corpo de referência orbital.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        center = camera.world_to_screen(position)
        body_radius = camera.world_to_screen_size(self.radius)
        ring_radius = body_radius + self.SELECTION_MARGIN

        renderer.draw_circle(
            color,
            center,
            ring_radius,
            width=2,
        )