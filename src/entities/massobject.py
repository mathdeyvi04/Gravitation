import math
import pygame
import random
from typing import Optional

from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity


class MassObject(Entity):
    """@brief Corpo com massa e raio que interage gravitacionalmente.

    @details Representa o estado físico de um corpo — posição, velocidade,
    massa, raio e cor — mas **não integra a si mesmo**. A integração é
    responsabilidade de um `Integrator` externo, que lê e escreve esses
    campos em passos fixos. Essa separação permite trocar o método
    numérico (Euler, RK4, Verlet) sem tocar no corpo.

    Campos auxiliares:

    - `previous_position`: estado imediatamente anterior ao último
      passo fixo, para interpolação suave no render.
    - `last_force`: força da última avaliação do integrador, usada
      apenas pelo `VectorVisualizer`.
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
        """@brief Cria um corpo a partir de `config`.

        @details Qualquer parâmetro `None` é sorteado dentro das faixas
        de `config`: posição uniforme no mundo; massa em `mass_range`;
        raio derivado da massa (ou `radius_range[0]` se o intervalo for
        degenerado); e velocidade isotrópica com módulo em
        `velocity_range` — direção e módulo separados, em vez de `vx`/`vy`
        independentes.

        @param config Configuração com as faixas de sorteio e o mundo.
        @param mass Massa; sorteada se None.
        @param radius Raio; derivado se None.
        @param position Posição inicial; sorteada se None.
        @param velocity Velocidade inicial; sorteada se None.
        @param color Cor do corpo.
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
            # Mapeia linearmente a massa em [0,1] e aplica raiz cúbica
            # para aproximar a relação r ∝ m^(1/3); sem essa derivação,
            # corpos de massas diferentes teriam densidades
            # discrepantes e a fusão pareceria arbitrária.
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
        # `velocity` já é não-None neste ponto (foi sorteada acima se
        # necessário); a cópia evita aliasing com o vetor do chamador.
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
        """@brief Aceleração adicional específica do corpo.

        @details No-op na base; subclasses como `Rocket` devolvem o
        empuxo aqui.

        @param position Posição avaliada pelo integrador.
        @param velocity Velocidade avaliada pelo integrador.
        @return `None` por padrão (sem aceleração extra).
        """
        return None

    def prediction_acceleration(
        self,
        elapsed_time: float,
    ) -> Optional[pygame.Vector2]:
        """@brief Aceleração adicional usada pela previsão futura.

        @param elapsed_time Tempo desde o início da predição.
        @return `None` por padrão (sem aceleração extra).
        """
        return None

    def update(self, delta_time: float) -> None:
        """@brief No-op: `MassObject` só reage a passos fixos.

        @details Manter este método vazio garante que o `update` de
        delta variável — usado pela `Application` para lógica de frame —
        não interfira na física.

        @param delta_time Tempo desde o último frame.
        """

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief No-op: a integração é feita externamente pelo `Integrator`.

        @details O integrador lê `position`, `velocity`,
        `previous_position` e `mass` e escreve nesses campos. A escolha
        do método numérico pertence ao integrador, não ao corpo.

        @param fixed_delta_time Passo fixo em segundos.
        """

    def collision_circles(
        self,
        alpha: float = 1.0,
    ) -> tuple[tuple[pygame.Vector2, float], ...]:
        """@brief Um único círculo centrado em `position` com raio `radius`.

        @param alpha Fator de interpolação em [0, 1].
        @return Tupla com um par `(centro, raio)` em world_space.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )
        return ((position, self.radius),)

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        """@brief Desenha o corpo como um disco preenchido.

        @details A posição de desenho é interpolada entre
        `previous_position` e `position` por `alpha`, suavizando o
        movimento quando a taxa de render difere da de passos fixos.
        Corpos fora da área visível são ignorados.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação em [0, 1].
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

    def merge_with(self, other: "MassObject") -> None:
        """@brief Absorve `other`, conservando massa, momento e volume.

        @details Centro de massa e velocidade resultantes são as médias
        ponderadas por massa — o que preserva o momento linear total do
        par. O raio é recalculado assumindo densidade uniforme
        (`r ∝ m^(1/3)`). `other` é destruído ao final.

        @param other Corpo a ser absorvido.
        """
        total = self.mass + other.mass
        if total <= 0.0:
            other.destroy()
            return

        # Pesos por massa: centro de massa e velocidade do CM.
        self.position.update(
            self.position * self.mass + other.position * other.mass
            / total
        )
        self.velocity.update(
            self.velocity * self.mass + other.velocity * other.mass
            / total
        )
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
        """@brief AABB do disco em world_space na posição interpolada.

        @param alpha Fator de interpolação em [0, 1].
        @return `(min_x, min_y, max_x, max_y)`.
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
        """@brief Indica se o ponto em world_space está dentro do disco.

        @details Usa a posição interpolada por `alpha`, coerente com o
        que está sendo desenhado no mesmo frame.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return `True` se o ponto está dentro do disco.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        # Comparação em quadrados: evita a raiz quadrada.
        return (
            position.distance_squared_to(world_position)
            <= self.radius * self.radius
        )

    def click_distance(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> float:
        """@brief Distância do ponto à superfície do disco (0.0 se dentro).

        @details Coerente com `hit_test` e `render`: usa a posição
        interpolada. É o que o `EntityPicker` consulta na fase de
        tolerância, quando nenhuma entidade contém o clique diretamente.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return Distância à superfície do disco, ou `0.0` se dentro.
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
        """@brief Desenha um anel por sub-círculo da forma de colisão.

        @details Para um `MassObject` comum é um único anel; para
        entidades com forma composta (ex.: `Rocket`), produz um
        contorno que coincide exatamente com a zona onde a fusão
        dispara.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação em [0, 1].
        @param color Cor do anel de seleção.
        """
        for center_world, radius_world in self.collision_circles(alpha):
            center = camera.world_to_screen(center_world)
            body_radius = camera.world_to_screen_size(radius_world)
            ring_radius = body_radius + self.SELECTION_MARGIN

            renderer.draw_circle(
                color,
                center,
                ring_radius,
                width=2,
            )