import math
import random
import pygame
from pathlib import Path
from typing import Optional, Union
from time import perf_counter
from src.config import Configs, SimulationConfig
from src.engine.application import Application
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
    """Simulação gravitacional de N corpos.

    A `World` guarda as entidades; esta classe define a ordem do passo
    físico — gravidade, integração, fusão, purga — e a política de
    fusão (maior absorve o menor).
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
    _STAR_COUNT_RANGE = (1, 2)
    _STAR_MASS_RANGE = (3_000_000.0, 6_000_000.0)
    _STAR_RADIUS_RANGE = (700.0, 1000.0)
    _STAR_ORBIT_RADIUS_RANGE = (1200.0, 2000.0)

    # --- Planetas -----------------------------------------------------
    _PLANET_COUNT_RANGE = (2, 5)
    _PLANET_MASS_RANGE = (20_000.0, 60_000.0)
    _PLANET_RADIUS_RANGE = (100.0, 140.0)
    _PLANET_BASE_ORBIT_FACTOR = 2.0
    _PLANET_ORBIT_STEP_FACTOR = 1.0

    # --- Luas ---------------------------------------------------------
    _MOON_COUNT_RANGE = (0, 2)
    _MOON_MASS_RANGE = (300.0, 800.0)
    _MOON_RADIUS_RANGE = (22.0, 35.0)
    _MOON_ORBIT_FACTOR_RANGE = (3.5, 6.0)

    # --- Meteoros -----------------------------------------------------
    _METEOR_COUNT_RANGE = (1, 3)
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
    _ROCKET_THRUST_ACCELERATION = 1500.0

    # Comprimento máximo, em pixels de tela, de uma seta de força.
    # Vale como teto de saturação: forças maiores que a referência do
    # corpo continuam desenhadas nesse tamanho.
    FORCE_VECTOR_SCREEN_LENGTH = 120.0

    # --- Câmera -------------------------------------------------------
    _INITIAL_ZOOM = 0.1

    @classmethod
    def _random_planet_color(cls) -> tuple[int, int, int]:
        return random.choice(cls._PLANET_PALETTE)

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
    ) -> None:
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

    def _register_mass_object_properties(self) -> None:
        """Registra as propriedades de `MassObject` exibíveis no inspector.

        Define quais campos aparecem na HUD quando um corpo é selecionado,
        na ordem em que são exibidos. Cada `InspectionProperty` combina um
        rótulo com um getter (extrai o valor do corpo) e um formatador
        (converte o valor em string).

        Chamado uma única vez, no `__init__`. Tipos que não são registrados
        aqui não aparecem no inspector mesmo quando selecionados.
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
        """Registra as propriedades entre dois MassObject."""

        self.relation_inspector.register(
            MassObject,
            MassObject,

            lambda primary, reference: (
                RelativeOrbit2D.from_bodies(
                    primary,
                    reference,
                    self.sim_config.gravitational_constant,
                )
            ),

            InspectionProperty(
                "Distância relativa",
                lambda orbit: orbit.distance,
                lambda value: f"{value:.2f}",
            ),

            InspectionProperty(
                "Vetor distância",
                lambda orbit: orbit.relative_position,
                lambda vector: (
                    f"({vector.x:.2f}, {vector.y:.2f})"
                ),
            ),

            InspectionProperty(
                "Velocidade relativa",
                lambda orbit: orbit.relative_speed,
                lambda value: f"{value:.2f}",
            ),

            InspectionProperty(
                "Vetor velocidade",
                lambda orbit: orbit.relative_velocity,
                lambda vector: (
                    f"({vector.x:.2f}, {vector.y:.2f})"
                ),
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
                "Δv de escape",
                lambda orbit: orbit.escape_delta_v,
                lambda vector: (
                    "—"
                    if vector is None
                    else f"({vector.x:.2f}, {vector.y:.2f})"
                ),
            ),
        )

    def _register_rocket_properties(self) -> None:
        """Registra propriedades específicas do foguete."""

        self.inspector.register(
            Rocket,

            InspectionProperty(
                "Ângulo",
                lambda rocket: math.degrees(
                    rocket.angle
                ),
                lambda value: f"{value:.2f}°",
            ),

            InspectionProperty(
                "Direção do bico",
                lambda rocket: rocket.forward,
                lambda vector: (
                    f"({vector.x:.2f}, "
                    f"{vector.y:.2f})"
                ),
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
        )

    # -- Internos -----------------------------------------------------

    def _seed_solar_system(self) -> None:
        """Gera um sistema estelar com hierarquia de corpos.

        Composição:
        - 1 a 3 estrelas massivas (mutuamente orbitantes se > 1);
        - 2 a 5 planetas por estrela, em órbitas circulares espaçadas;
        - 0 a 2 luas por planeta, orbitando o planeta;
        - 4 a 8 meteoros em órbitas muito excêntricas ao redor de
          estrelas ou planetas;
        - 1 foguete acoplado à superfície de um planeta sorteado.

        As massas e raios decrescem por ordens de magnitude entre as
        classes para que cada corpo seja gravitacionalmente dominado
        pelo seu hospedeiro.
        """
        cx = self.sim_config.world_width * 0.5
        cy = self.sim_config.world_height * 0.5

        stars = self._spawn_stars(cx, cy)

        planets: list[MassObject] = []
        for star in stars:
            planets.extend(self._spawn_planet_system(star))

        self._spawn_meteors(stars, planets)

        # Foguete nasce acoplado a um planeta sorteado.
        host = random.choice(planets)
        self._spawn_rocket(host)

    def _spawn_stars(
        self,
        cx: float,
        cy: float,
    ) -> list[MassObject]:
        """Sorteia 1 a 3 estrelas massivas.

        Com uma única estrela, ela fica em repouso no centro do mundo.
        Com múltiplas, todas são dispostas em polígono regular ao redor
        do centro e recebem velocidade tangencial para orbitar o
        baricentro.
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
        """Sorteia planetas em órbitas circulares ao redor de `star`.

        A órbita base está a 2 raios estelares do centro e cada
        planeta subsequente avança 1 raio estelar, então os planetas
        ficam claramente separados na nova escala.
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
        """Sorteia uma lua orbitando `planet`."""
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
        """Sorteia poucos meteoros em órbitas muito excêntricas.

        Cada meteoro é colocado no **afélio** de uma órbita com
        excentricidade entre 0.6 e 0.9 em torno de uma estrela ou
        planeta sorteado. A velocidade no afélio é perpendicular ao
        raio e tem módulo `sqrt(G·M/r) · sqrt(1 − e)`, condição exata
        para que aquele ponto seja o afélio da elipse de excentricidade
        `e`. O resultado são trajetórias que mergulham no sistema e
        voltam a se afastar, com dinâmica visualmente interessante.
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

    def _spawn_rocket(self, host: MassObject) -> None:
        """Cria o foguete acoplado à superfície de `host`."""
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
        self.rocket.dock_to(host, angle)

    def _resolve_merges(self) -> None:
        """Funde pares sobrepostos que estejam se aproximando rápido.

        Pares cuja velocidade relativa é menor que
        `merge_min_relative_speed` são ignorados — é o que permite o
        foguete repousar sobre um planeta sem ser absorvido.
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

                relative_speed = (a.velocity - b.velocity).length()
                if relative_speed < min_speed:
                    j += 1
                    continue

                if a.mass >= b.mass:
                    a.merge_with(b)
                else:
                    b.merge_with(a)
                    a = b
                j = i + 1
            i += 1

    # -- Ciclo de vida -------------------------------------------------

    def update(self, delta_time: float) -> None:
        """Lógica de delta variável: quando algo variar no sistema."""

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
        """Executa um passo físico completo usando RK4."""
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

        for body, acceleration in zip(bodies, accelerations):
            body.last_force.update(acceleration * body.mass)

        self._resolve_merges()
        self.world.purge_inactive()

    def render(self, renderer) -> None:
        """Desenha a `World` com o alpha de interpolação atual."""
        alpha = self.clock.interpolation_alpha

        force_time = 0.0
        if self.force_vectors.enabled:
            start_force = perf_counter()

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

            force_time = perf_counter() - start_force

        selected = self.inspector.selected
        reference: AnyGrav = self.relation_inspector.reference
        trajectory_focus = (
            (selected,)
            if isinstance(
                selected,
                Rocket,
            )
            else ()
        )
        trajectory_time = 0
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
            f"Pressione T para {trajectory_action} a trajetória futura{': {:.2f}ms'.format(trajectory_time * 1000) if self.future_trajectory.enabled else ''}",
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


def main() -> None:
    config = Configs(
        width=1500,
        height=1000,
        title="Gravitation",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    sim_config = SimulationConfig()
    Gravitation(config, sim_config).run()


if __name__ == "__main__":
    main()