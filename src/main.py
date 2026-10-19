import math
import random
import pygame
import argparse
from pathlib import Path
from typing import Optional, Union
from time import perf_counter
from src.config import Configs, SimulationConfig
from src.engine.application import Application
from src.engine.memory_monitor import MemoryMonitor
from src.engine.vector_visualizer import VectorVisualizer
from src.engine.trajectory_visualizer import TrajectoryVisualizer
from src.engine.world import World
from src.entities.massobject import MassObject
from src.physics.gravity import NewtonianGravity
from src.physics.rk4 import RK4Integrator
from src.entities.entity_picker import EntityPicker
from src.physics.relative_orbit import RelativeOrbit2D
from src.physics.collision import collides
from src.engine.inspector import (
    AnyGrav,
    Inspector,
    InspectorHUD,
    InspectionProperty,
    RelationInspector,
)
from src.entities.rocket import Rocket
from src.physics.composite_acceleration import (
    CompositeAccelerationModel,
)

class Gravitation(Application):
    """@brief Simulação gravitacional de N corpos.

    @details A `World` guarda as entidades; esta classe define a ordem
    do passo físico — gravidade, integração, fusão, purga — e a
    política de fusão (maior absorve o menor).
    """

    _PLANET_PALETTE = (
        (100, 150, 220),
        (200, 120, 80),
        (140, 200, 140),
        (200, 180, 120),
        (180, 100, 140),
        (110, 200, 200),
    )

    # --- Estrelas -----------------------------------------------------
    _STAR_COUNT_RANGE = (1, 1)
    _STAR_MASS_RANGE = (1_000_000.0, 2_000_000.0)
    _STAR_RADIUS_RANGE = (700.0, 1000.0)
    _STAR_ORBIT_RADIUS_RANGE = (1200.0, 2000.0)

    # --- Planetas -----------------------------------------------------
    _PLANET_COUNT_RANGE = (0, 0)
    _PLANET_MASS_RANGE = (20_000.0, 60_000.0)
    _PLANET_RADIUS_RANGE = (100.0, 140.0)
    _PLANET_BASE_ORBIT_FACTOR = 4.0
    _PLANET_ORBIT_STEP_FACTOR = 1.0

    # --- Luas ---------------------------------------------------------
    _MOON_COUNT_RANGE = (0, 0)
    _MOON_MASS_RANGE = (300.0, 800.0)
    _MOON_RADIUS_RANGE = (22.0, 35.0)
    _MOON_ORBIT_FACTOR_RANGE = (3.5, 6.0)

    # --- Meteoros -----------------------------------------------------
    _METEOR_COUNT_RANGE = (0, 0)
    _METEOR_MASS_RANGE = (10.0, 40.0)
    _METEOR_RADIUS_RANGE = (4.0, 8.0)
    _METEOR_ECCENTRICITY_RANGE = (0.6, 0.9)
    _METEOR_STAR_APHELION_FACTOR = (4.0, 10.0)
    _METEOR_PLANET_APHELION_FACTOR = (3.0, 6.0)

    # --- Foguete ------------------------------------------------------
    _ROCKET_BODY_LENGTH = 16.0
    _ROCKET_BODY_WIDTH = 7.0
    _ROCKET_NOSE_LENGTH = 9.0
    _ROCKET_MASS = 40.0
    _ROCKET_THRUST_ACCELERATION = 1000.0
    _WIDGET_AXIS_COLOR = (70, 70, 85)

    # Comprimento máximo, em pixels de tela, de uma seta de força.
    # Vale como teto de saturação: forças maiores que a referência do
    # corpo continuam desenhadas nesse tamanho.
    FORCE_VECTOR_SCREEN_LENGTH = 120.0

    # --- Câmera -------------------------------------------------------
    _INITIAL_ZOOM = 0.1

    @classmethod
    def _random_planet_color(cls) -> tuple[int, int, int]:
        """@brief Sorteia uma cor da paleta de planetas.

        @return Tupla RGB escolhida aleatoriamente em `_PLANET_PALETTE`.
        """
        return random.choice(cls._PLANET_PALETTE)

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        memory_monitor: MemoryMonitor | None = None,
    ) -> None:
        """@brief Inicializa a cena, subsistemas e o sistema estelar.

        @details Configura gravidade, integrador, visualizadores e
        inspector; popula o mundo com `_seed_solar_system`; centraliza
        a câmera no foguete; e, se fornecido, grava um snapshot inicial
        no monitor de memória.

        @param config Configuração da janela e do loop.
        @param sim_config Configuração do mundo e da física.
        @param memory_monitor Monitor opcional de memória.
        """
        super().__init__(config)
        self.sim_config = sim_config
        self.world: World[Union[MassObject, Rocket]] = World()
        self.rocket: Optional[Rocket] = None

        self.gravity = NewtonianGravity(
            gravitational_constant=sim_config.gravitational_constant,
            softening=sim_config.gravitational_softening,
        )
        self.integrator = RK4Integrator()
        # Estarão em ms
        self.integration_time_last_step = 0.0

        # Inicializamos as features de visualizações
        self.force_vectors = VectorVisualizer(
            color=(80, 220, 255)
        )
        self.future_trajectory = TrajectoryVisualizer(
            steps=sim_config.future_trajectory_steps,
            color=(170, 170, 255),
        )
        self.entity_picker = EntityPicker()
        self.inspector = Inspector()
        self.inspector_hud = InspectorHUD()
        self._register_mass_object_properties()
        self.relation_inspector = RelationInspector()
        self._register_mass_object_relation_properties()
        self._register_rocket_properties()

        # Iniciamos o sistema
        self._seed_solar_system()

        # A câmera nasce centrada no foguete — o jogador abre a cena
        # olhando diretamente para o planeta em que ele está acoplado.
        # O zoom inicial é suficientemente afastado para ver o planeta
        # inteiro mais um pouco de espaço; o jogador ajusta com o scroll.
        if self.rocket is not None:
            self.camera.set_position(self.rocket.position)
        self.camera.set_zoom(self._INITIAL_ZOOM)

        # Quando True, a câmera centraliza a cada frame no corpo
        # selecionado. Ativado/desativado com a tecla V.
        self.following = False

        self.memory_monitor = memory_monitor
        if self.memory_monitor is not None:
            # Baseline imediato: sempre existe um snapshot #0 para diffs.
            self.memory_monitor.capture(
                label="startup",
                context={"phase": "init"},
            )

    def _register_mass_object_properties(self) -> None:
        """@brief Registra as propriedades de `MassObject` no inspector.

        @details Define massa, raio, módulo e vetor da velocidade como
        campos exibíveis, na ordem em que aparecem no painel. Chamado
        uma única vez no `__init__`; tipos não registrados não aparecem
        no inspector mesmo quando selecionados.
        """
        self.inspector.register(
            MassObject,
            InspectionProperty(
                "Massa",
                lambda body: body.mass,
                lambda value: f"{value:.2f}",
            ),
            InspectionProperty(
                "Raio",
                lambda body: body.radius,
                lambda value: f"{value:.2f}",
            ),
            InspectionProperty(
                "Velocidade",
                lambda body: body.velocity.length(),
                lambda value: f"{value:.2f}",
            ),
            InspectionProperty(
                "Vetor velocidade",
                lambda body: body.velocity,
                lambda vector: f"({vector.x:.2f}, {vector.y:.2f})",
            ),
        )

    def _register_mass_object_relation_properties(self) -> None:
        """@brief Registra as propriedades relacionais entre `MassObject`.

        @details Expõe elementos orbitais relativos (calculados por
        `RelativeOrbit2D.from_bodies`) e, quando o corpo principal é um
        `Rocket`, os coeficientes de acoplagem e desacoplagem em
        relação ao corpo de referência.
        """
        self.relation_inspector.register(
            MassObject,
            MassObject,

            lambda primary, reference: (
                RelativeOrbit2D.from_bodies(
                    reference,
                    primary,
                    self.sim_config.gravitational_constant,
                )
            ),

            InspectionProperty(
                "Distância",
                lambda orbit: orbit.distance,
                lambda value: f"{value:.2f}",
            ),

            InspectionProperty(
                "Velocidade relativa",
                lambda orbit: (
                    -orbit.relative_speed
                    if orbit.radial_velocity < 0.0
                    else orbit.relative_speed
                ),
                lambda value: f"{value:+.2f}",
            ),

            InspectionProperty(
                "Velocidade radial",
                lambda orbit: orbit.radial_velocity,
                lambda value: f"{value:+.2f}",
            ),

            InspectionProperty(
                "Velocidade tangencial",
                lambda orbit: orbit.tangential_velocity,
                lambda value: f"{value:+.2f}",
            ),

            InspectionProperty(
                "Excentricidade",
                lambda orbit: orbit.eccentricity,
                lambda value: (
                    "Indefinida"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            InspectionProperty(
                "Tipo de órbita",
                lambda orbit: orbit.orbit_type,
                lambda value: value,
            ),

            InspectionProperty(
                "Menor distância ao foco",
                lambda orbit: orbit.periapsis_distance,
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            InspectionProperty(
                "Maior distância ao foco",
                lambda orbit: orbit.apoapsis_distance,
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            InspectionProperty(
                "Semi-eixo maior",
                lambda orbit: orbit.semi_major_axis,
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            InspectionProperty(
                "Período orbital",
                lambda orbit: orbit.orbital_period,
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}s"
                ),
            ),

            InspectionProperty(
                "Módulo Δv de escape",
                lambda orbit: (
                    None
                    if orbit.escape_delta_v is None
                    else orbit.escape_delta_v.length()
                ),
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            InspectionProperty(
                "Δv radial de escape",
                lambda orbit: orbit.escape_delta_v_radial,
                lambda value: "—" if value is None else f"{value:+.2f}",
            ),

            InspectionProperty(
                "Δv tangencial de escape",
                lambda orbit: orbit.escape_delta_v_tangential,
                lambda value: "—" if value is None else f"{value:+.2f}",
            ),

            # O getter abaixo roda *em cada frame* enquanto o par
            # (foguete, referência) estiver selecionado e executa a
            # simulação de subida em `detach_coeficient` — custo
            # considerável. Ver `DESIGN.md §5`.
            InspectionProperty(
                f"Desacoplagem",
                lambda _orbit: (
                    self.inspector.selected.detach_coeficient(
                        self.relation_inspector.reference,
                        body_lengths=3,
                        gravitational_constant=self.sim_config.gravitational_constant,
                        fixed_timestep=self.config.fixed_timestep,
                    )
                    if isinstance(self.inspector.selected, Rocket)
                    else None
                ),
                lambda value: "INF" if value is None else f"{value * self.config.fixed_timestep:.2f}s",
            ),

            # Idem para `attach_coeficient`: duas fases (coast + burn)
            # com limites de passos, custo alto por frame.
            InspectionProperty(
                "Acoplagem",
                lambda _orbit: (
                    self.inspector.selected.attach_coeficient(
                        self.relation_inspector.reference,
                        gravitational_constant=self.sim_config.gravitational_constant,
                        landing_max_speed=self.sim_config.landing_max_speed,
                        fixed_timestep=self.config.fixed_timestep,
                    )
                    if isinstance(self.inspector.selected, Rocket)
                    else None
                ),
                lambda value: (
                    "—"
                    if value is None
                    else f"{value * self.config.fixed_timestep:.2f}s"
                ),
            ),
        )

    def _register_rocket_properties(self) -> None:
        """@brief Registra as propriedades específicas do `Rocket`.

        @details Expõe ângulo, estado de propulsão e fração de
        combustível restante, com os widgets de orientação e de tanque
        anexados.
        """
        self.inspector.register(
            Rocket,

            InspectionProperty(
                "Ângulo",
                lambda rocket: -math.degrees(rocket.angle),
                lambda value: f"{value:.2f}°",
                widget=self._draw_rocket_widget,
            ),

            InspectionProperty(
                "Propulsão",
                lambda rocket: rocket.thrusting,
                lambda value: (
                    "ATIVA"
                    if value
                    else "INATIVA"
                ),
            ),

            InspectionProperty(
                "Combustível",
                lambda rocket: (
                    rocket.fuel / rocket.fuel_max
                    if rocket.fuel_max > 0.0
                    else 0.0
                ),
                lambda fraction: f"{100.0 * fraction:.1f}%",
                widget=self._draw_fuel_widget,
            ),
        )

    def _draw_rocket_widget(
        self,
        rocket,
        renderer,
        rect,
    ) -> None:
        """@brief Desenha o ícone orientado do foguete no inspector.

        @details Escala o foguete para caber em `rect`, desenha os eixos
        X/Y de referência e projeta o corpo e o bico com a rotação
        atual. A orientação usa `rocket.angle` diretamente — o mesmo
        referencial de `Rocket.render` — para o ícone nunca contradizer
        a cena.

        @param rocket Foguete cuja orientação será exibida.
        @param renderer Renderizador alvo.
        @param rect Retângulo onde o widget é desenhado.
        """
        total_length = rocket.body_length + rocket.nose_length
        available = min(rect.width, rect.height) - 8

        if available <= 0 or total_length <= 0:
            return

        cx = rect.x + rect.width * 0.5
        cy = rect.y + rect.height * 0.5

        # Referencial de direção: linhas nos eixos X e Y, com margem
        # curta nas pontas para não colar nas bordas do rect.
        margin = 4
        renderer.draw_line(
            self._WIDGET_AXIS_COLOR,
            (rect.x + margin, cy),
            (rect.right - margin, cy),
            width=1,
        )
        renderer.draw_line(
            self._WIDGET_AXIS_COLOR,
            (cx, rect.y + margin),
            (cx, rect.bottom - margin),
            width=1,
        )

        scale = available / total_length

        body_l = rocket.body_length * scale
        body_w = rocket.body_width * scale
        nose_l = rocket.nose_length * scale

        cos_a = math.cos(rocket.angle)
        sin_a = math.sin(rocket.angle)

        half_l = body_l * 0.5
        half_w = body_w * 0.5

        # Transforma coordenadas locais (eixo do foguete) em
        # screen_space com a rotação atual.
        def transform(lx: float, ly: float) -> tuple[float, float]:
            return (
                cx + lx * cos_a - ly * sin_a,
                cy + lx * sin_a + ly * cos_a,
            )

        body_points = (
            transform(-half_l, -half_w),
            transform(half_l, -half_w),
            transform(half_l, half_w),
            transform(-half_l, half_w),
        )
        nose_points = (
            transform(half_l, -half_w),
            transform(half_l + nose_l, 0.0),
            transform(half_l, half_w),
        )

        renderer.draw_polygon(rocket.color, body_points)
        renderer.draw_polygon(rocket.nose_color, nose_points)

    def _draw_fuel_widget(self, rocket, renderer, rect) -> None:
        """@brief Desenha a barra de combustível do foguete.

        @details Borda fixa, fundo escuro e preenchimento proporcional
        à fração restante. A cor muda com o nível: verde acima de 50%,
        amarelo entre 25% e 50%, vermelho abaixo de 25%.

        @param rocket Foguete cujo tanque será exibido.
        @param renderer Renderizador alvo.
        @param rect Retângulo onde o widget é desenhado.
        """
        margin = 4
        bar = pygame.Rect(
            rect.x + margin,
            rect.y + margin,
            rect.width - margin * 2,
            rect.height - margin * 2,
        )

        renderer.draw_rect((20, 20, 28), bar)

        fraction = (
            rocket.fuel / rocket.fuel_max
            if rocket.fuel_max > 0.0
            else 0.0
        )
        fraction = max(0.0, min(1.0, fraction))

        if fraction > 0.5:
            fill_color = (80, 200, 80)
        elif fraction > 0.25:
            fill_color = (220, 200, 60)
        else:
            fill_color = (220, 80, 60)

        fill_width = int(bar.width * fraction)
        if fill_width > 0:
            fill = pygame.Rect(bar.x, bar.y, fill_width, bar.height)
            renderer.draw_rect(fill_color, fill)

        renderer.draw_rect((180, 180, 180), bar, width=1)

    # -- Internos -----------------------------------------------------

    def _seed_solar_system(self) -> None:
        """@brief Gera o sistema estelar completo.

        @details Cria estrelas, planetas, luas, meteoros e o foguete
        acoplado a um planeta sorteado (ou à primeira estrela, se não
        houver planetas). As massas e raios decrescem por ordens de
        magnitude entre as classes para que cada corpo seja
        gravitacionalmente dominado pelo seu hospedeiro.
        """
        cx = self.sim_config.world_width * 0.5
        cy = self.sim_config.world_height * 0.5

        stars = self._spawn_stars(cx, cy)

        planets: list[MassObject] = []
        for star in stars:
            planets.extend(self._spawn_planet_system(star))

        self._spawn_meteors(stars, planets)

        # Foguete nasce acoplado a um planeta sorteado.
        host = random.choice(planets) if len(planets) != 0 else stars[0]
        self._spawn_rocket(host)

    def _spawn_stars(
        self,
        cx: float,
        cy: float,
    ) -> list[MassObject]:
        """@brief Sorteia e posiciona as estrelas do sistema.

        @details Com uma única estrela, ela fica em repouso no centro do
        mundo. Com múltiplas, dispõe-as em polígono regular ao redor do
        centro e atribui velocidade tangencial para orbitar o baricentro.

        @param cx Coordenada X do centro do mundo.
        @param cy Coordenada Y do centro do mundo.
        @return Lista das estrelas criadas.
        """
        G = self.sim_config.gravitational_constant
        count = random.randint(*self._STAR_COUNT_RANGE)

        if count == 1:
            star = MassObject(
                config=self.sim_config,
                mass=random.uniform(*self._STAR_MASS_RANGE),
                radius=random.uniform(*self._STAR_RADIUS_RANGE),
                position=pygame.Vector2(cx, cy),
                velocity=pygame.Vector2(0.0, 0.0),
                color=(255, 200, 80),
            )
            self.world.add(star)
            return [star]

        orbit_radius = random.uniform(*self._STAR_ORBIT_RADIUS_RANGE)
        star_mass = random.uniform(*self._STAR_MASS_RANGE)
        star_radius = random.uniform(*self._STAR_RADIUS_RANGE)
        total_mass = count * star_mass

        # Velocidade orbital de N corpos iguais num anel de raio
        # `orbit_radius` ao redor do baricentro (ver problema clássico
        # do polígono regular).
        speed = math.sqrt(
            G * total_mass / (2.0 * count * orbit_radius)
        )

        stars: list[MassObject] = []
        phase = random.uniform(0.0, 2.0 * math.pi)

        for i in range(count):
            angle = phase + 2.0 * math.pi * i / count

            px = cx + orbit_radius * math.cos(angle)
            py = cy + orbit_radius * math.sin(angle)

            vx = -speed * math.sin(angle)
            vy = speed * math.cos(angle)

            star = MassObject(
                config=self.sim_config,
                mass=star_mass * random.uniform(0.9, 1.1),
                radius=star_radius * random.uniform(0.92, 1.08),
                position=pygame.Vector2(px, py),
                velocity=pygame.Vector2(vx, vy),
                color=(255, 200, 80),
            )
            self.world.add(star)
            stars.append(star)

        return stars

    def _spawn_planet_system(
        self,
        star: MassObject,
    ) -> list[MassObject]:
        """@brief Sorteia planetas em órbitas circulares ao redor de `star`.

        @details A órbita base fica a `_PLANET_BASE_ORBIT_FACTOR` raios
        estelares do centro; cada planeta subsequente avança
        `_PLANET_ORBIT_STEP_FACTOR` raio estelar. Cada planeta pode
        receber luas via `_spawn_moon`.

        @param star Estrela hospedeira.
        @return Lista dos planetas criados.
        """
        G = self.sim_config.gravitational_constant
        planet_count = random.randint(*self._PLANET_COUNT_RANGE)

        base_orbit = star.radius * self._PLANET_BASE_ORBIT_FACTOR
        step = star.radius * self._PLANET_ORBIT_STEP_FACTOR

        planets: list[MassObject] = []

        for i in range(planet_count):
            orbit_radius = (
                base_orbit
                + i * step
                + random.uniform(-0.15, 0.15) * step
            )
            angle = random.uniform(0.0, 2.0 * math.pi)

            px = star.position.x + orbit_radius * math.cos(angle)
            py = star.position.y + orbit_radius * math.sin(angle)

            # Órbita circular em torno de `star`, com jitter de ±10%
            # para não ficar exatamente periódica.
            speed = math.sqrt(G * star.mass / orbit_radius)
            speed *= random.uniform(0.9, 1.1)

            vx = star.velocity.x - speed * math.sin(angle)
            vy = star.velocity.y + speed * math.cos(angle)

            planet = MassObject(
                config=self.sim_config,
                mass=random.uniform(*self._PLANET_MASS_RANGE),
                radius=random.uniform(*self._PLANET_RADIUS_RANGE),
                position=pygame.Vector2(px, py),
                velocity=pygame.Vector2(vx, vy),
                color=self._random_planet_color(),
            )
            self.world.add(planet)
            planets.append(planet)

            moon_count = random.randint(*self._MOON_COUNT_RANGE)
            for _ in range(moon_count):
                self._spawn_moon(planet)

        return planets

    def _spawn_moon(self, planet: MassObject) -> None:
        """@brief Sorteia uma lua orbitando `planet`.

        @details Coloca a lua em órbita circular com raio em
        `_MOON_ORBIT_FACTOR_RANGE` vezes o raio do planeta e velocidade
        ajustada por um fator aleatório de 0.9 a 1.1.

        @param planet Planeta hospedeiro.
        """
        G = self.sim_config.gravitational_constant

        orbit_radius = (
            planet.radius
            * random.uniform(*self._MOON_ORBIT_FACTOR_RANGE)
        )
        angle = random.uniform(0.0, 2.0 * math.pi)

        px = planet.position.x + orbit_radius * math.cos(angle)
        py = planet.position.y + orbit_radius * math.sin(angle)

        speed = math.sqrt(G * planet.mass / orbit_radius)
        speed *= random.uniform(0.9, 1.1)

        vx = planet.velocity.x - speed * math.sin(angle)
        vy = planet.velocity.y + speed * math.cos(angle)

        moon = MassObject(
            config=self.sim_config,
            mass=random.uniform(*self._MOON_MASS_RANGE),
            radius=random.uniform(*self._MOON_RADIUS_RANGE),
            position=pygame.Vector2(px, py),
            velocity=pygame.Vector2(vx, vy),
            color=(185, 185, 200),
        )
        self.world.add(moon)

    def _spawn_meteors(
        self,
        stars: list[MassObject],
        planets: list[MassObject],
    ) -> None:
        """@brief Sorteia meteoros em órbitas muito excêntricas.

        @details Cada meteoro nasce no afélio de uma elipse de
        excentricidade em `_METEOR_ECCENTRICITY_RANGE`, ao redor de um
        planeta (60% de chance) ou estrela (40%). A velocidade no
        afélio é perpendicular ao raio e vale
        `sqrt(G·M/r) · sqrt(1 − e)` — condição exata para que o ponto
        seja o afélio da elipse de excentricidade `e`.

        @param stars Estrelas candidatas a hospedeiro.
        @param planets Planetas candidatos a hospedeiro.
        """
        G = self.sim_config.gravitational_constant
        count = random.randint(*self._METEOR_COUNT_RANGE)

        for _ in range(count):
            # 60% chance de orbitar um planeta, 40% uma estrela.
            if planets and random.random() < 0.6:
                host = random.choice(planets)
                factor = random.uniform(
                    *self._METEOR_PLANET_APHELION_FACTOR
                )
            else:
                host = random.choice(stars)
                factor = random.uniform(
                    *self._METEOR_STAR_APHELION_FACTOR
                )

            aphelion = host.radius * factor
            eccentricity = random.uniform(
                *self._METEOR_ECCENTRICITY_RANGE
            )
            angle = random.uniform(0.0, 2.0 * math.pi)

            px = host.position.x + aphelion * math.cos(angle)
            py = host.position.y + aphelion * math.sin(angle)

            circular_speed = math.sqrt(G * host.mass / aphelion)
            aphelion_speed = circular_speed * math.sqrt(1.0 - eccentricity)

            vx = host.velocity.x - aphelion_speed * math.sin(angle)
            vy = host.velocity.y + aphelion_speed * math.cos(angle)

            meteor = MassObject(
                config=self.sim_config,
                mass=random.uniform(*self._METEOR_MASS_RANGE),
                radius=random.uniform(*self._METEOR_RADIUS_RANGE),
                position=pygame.Vector2(px, py),
                velocity=pygame.Vector2(vx, vy),
                color=(150, 150, 150),
            )
            self.world.add(meteor)

    def _spawn_rocket(self, host: MassObject | None) -> None:
        """@brief Cria o foguete e o acopla a `host`, se fornecido.

        @details O foguete nasce em repouso na origem; se `host` é
        fornecido, `dock_to` o reposiciona na superfície do hospedeiro
        e aplica a colisão inelástica de acoplamento.

        @param host Corpo ao qual o foguete será acoplado; `None`
        deixa-o livre na origem.
        """
        angle = random.uniform(0.0, 2.0 * math.pi)

        self.rocket = Rocket(
            config=self.sim_config,
            mass=self._ROCKET_MASS,
            position=pygame.Vector2(0.0, 0.0),
            velocity=pygame.Vector2(0.0, 0.0),
            angle=angle,
            body_length=self._ROCKET_BODY_LENGTH,
            body_width=self._ROCKET_BODY_WIDTH,
            nose_length=self._ROCKET_NOSE_LENGTH,
            thrust_acceleration=self._ROCKET_THRUST_ACCELERATION,
            color=(220, 220, 230),
            nose_color=(240, 110, 80),
        )
        self.world.add(self.rocket)
        if host is not None:
            self.rocket.dock_to(host, angle)

    def _resolve_merges(self) -> None:
        """@brief Funde pares sobrepostos que estejam se aproximando rápido.

        @details Pares cuja velocidade relativa é menor que
        `merge_min_relative_speed` são ignorados — é o que permite o
        foguete repousar sobre um planeta sem ser absorvido. O corpo
        mais massivo absorve o outro via `merge_with`. Quando `b` é
        maior que `a`, o par é invertido e o laço recomeça a partir
        do absorvedor (agora em `a`).

        @details A fusão é feita por uma varredura O(N²) em pares
        ainda ativos. Pares cujo lado já foi absorvido por uma fusão
        anterior no mesmo passo são ignorados naturalmente por
        `not b.active` na iteração seguinte.
        """
        bodies = self.world.entities
        n = len(bodies)
        min_speed = self.sim_config.merge_min_relative_speed

        i = 0
        while i < n:
            a = bodies[i]
            if not a.active:
                i += 1
                continue
            j = i + 1
            while j < n:
                b = bodies[j]
                if not b.active or b is a:
                    j += 1
                    continue
                if not collides(a, b):
                    j += 1
                    continue

                # Aproximação lenta = "contato suave" (pouso), não fusão.
                relative_speed = (a.velocity - b.velocity).length()
                if relative_speed < min_speed:
                    j += 1
                    continue

                if a.mass >= b.mass:
                    a.merge_with(b)
                else:
                    b.merge_with(a)
                    # `a` foi absorvido; o sobrevivente é `b`.
                    # Reatribuímos para que o laço interno recomece
                    # a partir dele (o absorvedor pode ter mudado de
                    # massa/posição e agora colidir com outros).
                    a = b
                j = i + 1
            i += 1

    # -- Ciclo de vida -------------------------------------------------

    def update(self, delta_time: float) -> None:
        """@brief Lógica de delta variável executada uma vez por frame.

        @details Trata, nesta ordem: captura de memória (se habilitada),
        zoom por scroll, pan da câmera, teclas F/T/V, entrada do
        foguete (rotação e empuxo), seleção de corpo principal (botão
        esquerdo), seleção de referência (botão direito) e follow da
        câmera.

        @param delta_time Tempo desde o último frame, em segundos.
        """
        if (
                self.memory_monitor is not None
                and self.memory_monitor.should_capture()
        ):
            self.memory_monitor.capture(
                label=f"t{self.clock.total_time:.0f}s",
                context={
                    "entities": len(self.world.entities),
                    "active_entities": sum(
                        1 for e in self.world.entities if e.active
                    ),
                    "fps": round(self.clock.fps, 1),
                    "total_time": round(self.clock.total_time, 2),
                    "zoom": round(self.camera.zoom, 4),
                    "trajectory": self.future_trajectory.enabled,
                    "forces": self.force_vectors.enabled,
                    "following": self.following,
                    "dumps_so_far": self.memory_monitor.count,
                },
            )

        # Zoom por scroll, ancorado no cursor. A base 1.1 dá um passo
        # suave por notch; `**` acumula linearmente quando há vários.
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(
                self.input.mouse_position,
                1.1 ** self.input.mouse_wheel,
            )

        # Pan: enquanto o botão esquerdo estiver pressionado, a câmera
        # segue o mouse. Dividir por `zoom` converte o deslocamento em
        # pixels para unidades de mundo, então o conteúdo "gruda" no
        # cursor independentemente do nível de zoom.
        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(
                -self.input.mouse_delta / self.camera.zoom
            )

        if self.input.was_pressed(pygame.K_f):
            self.force_vectors.toggle()

        if self.input.was_pressed(pygame.K_t):
            self.future_trajectory.toggle()

        # Toggle do "seguir": desliga se estava ligado, liga se estava
        # desligado E existe algo selecionado. A expressão abaixo
        # combina os dois casos numa única atribuição booleana.
        if self.input.was_pressed(pygame.K_v):
            self.following = (
                not self.following
                and self.inspector.selected is not None
            )

        if (
            self.rocket is not None
            and self.rocket.active
        ):
            rotation_input = (
                int(
                    self.input.is_down(
                        pygame.K_RIGHT
                    )
                )
                -
                int(
                    self.input.is_down(
                        pygame.K_LEFT
                    )
                )
            )

            self.rocket.set_control(
                rotation_input,
                self.input.is_down(
                    pygame.K_SPACE
                ),
            )

        if self.input.was_mouse_button_clicked(
                pygame.BUTTON_LEFT
        ):
            selected = self.entity_picker.pick(
                self.input.mouse_position,
                self.camera,
                self.world.entities,
                alpha=1.0,
            )

            if selected is None:
                self.inspector.clear()
            else:
                self.inspector.select(selected)

            # Nova seleção principal invalida a relação anterior.
            self.relation_inspector.clear()


        elif self.input.was_mouse_button_pressed(
                pygame.BUTTON_RIGHT
        ):
            # O botão direito só funciona após existir
            # um corpo principal.
            primary = self.inspector.selected

            if primary is not None:

                reference = self.entity_picker.pick(
                    self.input.mouse_position,
                    self.camera,
                    self.world.entities,
                    alpha=1.0,
                    exclude=primary,
                )

                if reference is not None:
                    self.relation_inspector.select_reference(
                        reference
                    )

        # Seguir o corpo selecionado com a câmera. Se a seleção sumiu
        # (corpo destruído por fusão, ou clique no vazio), o modo é
        # desligado automaticamente.
        if self.following:
            target = self.inspector.selected
            if target is None:
                self.following = False
            else:
                self.camera.set_position(target.position)

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Executa um passo físico completo usando RK4.

        @details Sequência: atualiza rotação/combustível, integra com
        `RK4Integrator`, aplica `post_integrate` em cada corpo
        (necessário para o foguete acoplado), converte a aceleração do
        último estágio em força visual (`last_force`), tenta acoplar o
        foguete, resolve fusões e purga entidades inativas.

        @param fixed_delta_time Passo fixo em segundos.
        """
        bodies = self.world.entities

        # Rotação do foguete e outros no-ops de passo fixo.
        self.world.fixed_update(fixed_delta_time)

        acceleration_model = CompositeAccelerationModel(
            self.gravity,
            bodies,
        )

        start = perf_counter()
        accelerations = self.integrator.step(
            bodies,
            fixed_delta_time,
            acceleration_model,
        )
        elapsed = (perf_counter() - start) * 1000
        self.integration_time_last_step = elapsed

        # Deixa cada entidade reajustar o próprio estado após a
        # integração — o foguete acoplado usa isso para continuar
        # grudado no hospedeiro.
        for body in bodies:
            body.post_integrate()

        # `last_force` é guardado como *força* (a·m), não aceleração,
        # porque é isso que o `VectorVisualizer` espera receber.
        for body, acceleration in zip(bodies, accelerations):
            body.last_force.update(acceleration)
            body.last_force *= body.mass

        if self.rocket is not None and self.rocket.active:
            self.rocket.try_land(
                bodies,
                self.sim_config.landing_max_speed,
            )

        self._resolve_merges()
        self.world.purge_inactive()

    def render(self, renderer) -> None:
        """@brief Desenha mundo, seleções, visualizadores e HUD.

        @details Usa `clock.interpolation_alpha` para interpolar a
        posição entre passos fixos, coerente com o que `Entity.render`
        desenha. Se um `Rocket` estiver selecionado, sua trajetória é
        incluída em `focus_entities` mesmo quando a visualização global
        está desligada.

        @param renderer Renderizador alvo.
        """
        alpha = self.clock.interpolation_alpha

        selected = self.inspector.selected
        reference: AnyGrav = self.relation_inspector.reference
        # Só rastreamos a trajetória de um foguete selecionado; outros
        # corpos seguem a flag global `future_trajectory.enabled`.
        trajectory_focus = (
            (selected,)
            if isinstance(
                selected,
                Rocket,
            )
            else ()
        )

        if selected is not None:
            selected.render_selection(
                renderer,
                self.camera,
                alpha,
                color=(255, 255, 255),
            )

        if (
                selected is not None
                and reference is not None
                and reference is not selected
        ):
            reference.render_selection(
                renderer,
                self.camera,
                alpha,
                color=(255, 190, 70),
            )

        force_time = 0.0
        if self.force_vectors.enabled:
            force_time = perf_counter()

            # Materializa a lista uma vez. O generator do world é
            # esgotável, e precisamos de duas passadas: uma para achar
            # a maior força, outra para desenhar. A lista guarda só
            # referências — custo ~8 bytes por corpo.
            visible = list(
                self.world.visible_entities(self.camera, alpha)
            )

            # Maior força entre os corpos visíveis. Comparamos quadrados
            # no laço (evita sqrt por corpo) e tiramos a raiz uma vez no
            # fim.
            max_force_sq = 0.0
            for body in visible:
                f_sq = body.last_force.length_squared()
                if f_sq > max_force_sq:
                    max_force_sq = f_sq

            if max_force_sq > 1e-12:
                # Raiz aplicada uma única vez. Divisor comum a todas as
                # setas: a maior força vira 120 px, as demais ficam
                # proporcionais.
                reference_scale = max_force_sq ** 0.5

                for body in visible:
                    position = (
                        body.previous_position.lerp(body.position, alpha)
                        if alpha < 1.0
                        else body.position
                    )

                    self.force_vectors.draw(
                        renderer,
                        self.camera,
                        position,
                        body.last_force,
                        reference_scale=reference_scale,
                        max_screen_length=self.FORCE_VECTOR_SCREEN_LENGTH,
                    )

            force_time = perf_counter() - force_time

        trajectory_time = 0.0
        if self.future_trajectory.enabled:
            trajectory_time = (
                self.future_trajectory.draw(
                    renderer,
                    self.camera,
                    self.world.entities,
                    self.gravity,
                    self.clock.fixed_timestep,
                    focus_entities=trajectory_focus,
                )
            )

        self.world.render(
            renderer,
            self.camera,
            alpha,
        )

        # ------------------------------------------------------
        # Apresentação de HUD

        self.inspector_hud.draw(
            renderer,
            self.inspector,
            self.relation_inspector,
        )

        force_action = (
            "ocultar"
            if self.force_vectors.enabled
            else "ver"
        )

        trajectory_action = (
            "ocultar"
            if self.future_trajectory.enabled
            else "ver"
        )

        follow_hint = (
            "Pressione V para parar de seguir"
            if self.following
            else "Pressione V para seguir o corpo selecionado"
        )

        renderer.draw_text(
            follow_hint,
            (5, self.config.height - 85),
            size=25,
            color=(255, 255, 255),
        )

        renderer.draw_text(
            f"Pressione F para {force_action} os vetores de força{': {:.2f}ms'.format(force_time * 1000) if self.force_vectors.enabled else ''}",
            (5, self.config.height - 55),
            size=25,
            color=(255, 255, 255),
        )

        renderer.draw_text(
            f"Pressione T para {trajectory_action} a trajetória futura{': {:.2f}ms'.format(trajectory_time * 1000) if trajectory_time != 0.0 else ''}",
            (5, self.config.height - 25),
            size=25,
            color=(255, 255, 255),
        )

        renderer.draw_text(
            (
                f"Tempo de Passo de Simulação: "
                f"{self.integration_time_last_step:.2f}ms"
            ),
            (self.config.width - 320, self.config.height - 25),
            size=25,
            color=(255, 255, 255),
        )

    def shutdown(self) -> None:
        """@brief Encerra subsistemas e grava o snapshot final.

        @details Se o monitor de memória existir e estiver ativo,
        captura um último snapshot e para o rastreamento. Em seguida
        delega para `Application.shutdown`.
        """
        if self.memory_monitor is not None:
            if self.memory_monitor.active:
                self.memory_monitor.capture(
                    label="shutdown",
                    context={
                        "phase": "shutdown",
                        "total_time": round(self.clock.total_time, 2),
                        "frame_count": self.clock.frame_count,
                    },
                )
            self.memory_monitor.stop()
        super().shutdown()

def main() -> None:
    """@brief Ponto de entrada: parseia argumentos e inicia a aplicação.

    @details Configura a janela e o `SimulationConfig`, instancia o
    `MemoryMonitor` quando `--monitor-memory` é passado e executa
    `Gravitation.run`.
    """
    parser = argparse.ArgumentParser(
        prog="gravitation",
        description=(
            "Simulação gravitacional de N corpos. "
            "Sem flags, inicia a cena sem nenhum custo de instrumentação."
        ),
    )

    parser.add_argument(
        "--monitor-memory",
        action="store_true",
        help=(
            "Liga o monitor de memória: tracemalloc é ativado e snapshots "
            "são gravados automaticamente em intervalos regulares. "
            "Sem esta flag, nenhum custo é pago."
        ),
    )

    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path(__file__).parent.parent / "mem_dumps",
        metavar="DIR",
        help="Diretório dos snapshots. Padrão: ./mem_dumps/",
    )

    parser.add_argument(
        "--dump-interval",
        type=float,
        default=1,
        metavar="SEGUNDOS",
        help=(
            "Intervalo, em segundos, entre capturas automáticas. "
            "Padrão: 10.0."
        ),
    )

    parser.add_argument(
        "--max-dumps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Número máximo de snapshots por execução. Sem limite se "
            "omitido. Útil para sessões longas."
        ),
    )

    parser.add_argument(
        "--traceback-depth",
        type=int,
        default=25,
        metavar="N",
        help=(
            "Frames de traceback guardados por alocação. Maior = mais "
            "preciso para localizar vazamentos, arquivos maiores. "
            "Padrão: 25."
        ),
    )

    args = parser.parse_args()

    config = Configs(
        width=1500,
        height=1000,
        title="Gravitation",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    sim_config = SimulationConfig()

    monitor = None
    if args.monitor_memory:
        monitor = MemoryMonitor(
            dump_dir=args.dump_dir,
            traceback_depth=args.traceback_depth,
            interval=args.dump_interval,
            max_dumps=args.max_dumps,
            start_immediately=True,
        )
        print(
            f"[mem] monitor ativo — snapshots a cada "
            f"{args.dump_interval:.1f}s em {monitor.session_dir}"
            + (f" (máx. {args.max_dumps})" if args.max_dumps else "")
        )

    Gravitation(config, sim_config, memory_monitor=monitor).run()

if __name__ == "__main__":
    main()