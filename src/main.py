import math
import random
import pygame
from pathlib import Path
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
from src.engine.inspector import (
    Inspector,
    InspectorHUD,
    InspectionProperty,
    RelationInspector,
    RelativeInspectionProperty,
)



class Gravitation(Application):
    """Simulação gravitacional de N corpos.

    A `World` guarda as entidades; esta classe define a ordem do passo
    físico — gravidade, integração, fusão, purga — e a política de
    fusão (maior absorve o menor).
    """

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
    ) -> None:
        super().__init__(config)
        self.sim_config = sim_config
        self.world: World[MassObject] = World()

        self.gravity = NewtonianGravity(
            gravitational_constant=sim_config.gravitational_constant,
            softening=sim_config.gravitational_softening,
        )
        self.integrator = RK4Integrator()
        # Estarão em ms
        self.integration_time_last_step = 0.0

        # A câmera inicia no centro do mundo, que é o ponto para onde
        # `MassObject` distribui os corpos aleatórios. Sem isso, os
        # corpos nascem em `[0, W] × [0, H]` enquanto a câmera olha
        # para `[-W/2, +W/2] × [-H/2, +H/2]`, e quase tudo fica fora
        # da tela.
        self.camera.set_position(
            (sim_config.world_width * 0.5, sim_config.world_height * 0.5)
        )

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

        # Iniciamos o sistema
        self._seed_solar_system()

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

            RelativeInspectionProperty(
                "Distância relativa",
                lambda orbit: orbit.distance,
                lambda value: f"{value:.2f}",
            ),

            RelativeInspectionProperty(
                "Vetor distância",
                lambda orbit: orbit.relative_position,
                lambda vector: (
                    f"({vector.x:.2f}, {vector.y:.2f})"
                ),
            ),

            RelativeInspectionProperty(
                "Velocidade relativa",
                lambda orbit: orbit.relative_speed,
                lambda value: f"{value:.2f}",
            ),

            RelativeInspectionProperty(
                "Vetor velocidade",
                lambda orbit: orbit.relative_velocity,
                lambda vector: (
                    f"({vector.x:.2f}, {vector.y:.2f})"
                ),
            ),

            RelativeInspectionProperty(
                "Excentricidade",
                lambda orbit: orbit.eccentricity,
                lambda value: (
                    "Indefinida"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            RelativeInspectionProperty(
                "Tipo de órbita",
                lambda orbit: orbit.orbit_type,
                lambda value: value,
            ),

            RelativeInspectionProperty(
                "Menor distância ao foco",
                lambda orbit: orbit.periapsis_distance,
                lambda value: (
                    "Indefinida"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            RelativeInspectionProperty(
                "Maior distância ao foco",
                lambda orbit: orbit.apoapsis_distance,
                lambda value: (
                    "Não definida"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            RelativeInspectionProperty(
                "Semi-eixo maior",
                lambda orbit: orbit.semi_major_axis,
                lambda value: (
                    "Indefinido"
                    if value is None
                    else f"{value:.2f}"
                ),
            ),

            RelativeInspectionProperty(
                "Período orbital",
                lambda orbit: orbit.orbital_period,
                lambda value: (
                    "—"
                    if value is None
                    else f"{value:.2f}s"
                ),
            ),

            RelativeInspectionProperty(
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

            RelativeInspectionProperty(
                "Δv de escape",
                lambda orbit: orbit.escape_delta_v,
                lambda vector: (
                    "—"
                    if vector is None
                    else f"({vector.x:.2f}, {vector.y:.2f})"
                ),
            ),
        )

    # -- Internos -----------------------------------------------------

    def _seed_solar_system(self) -> None:
        """Corpo central massivo com orbitadores circulares ao redor.

        Cada orbitador recebe velocidade tangencial `v = sqrt(G·M / r)`,
        condição de órbita circular para um corpo de teste. Massas dos
        orbitadores são ordens de magnitude menores que a central, para
        que a dinâmica seja dominada pelo corpo massivo.
        """
        cx = self.sim_config.world_width * 0.5
        cy = self.sim_config.world_height * 0.5

        central_mass = 5000.0
        self.world.add(MassObject(
            config=self.sim_config,
            mass=central_mass,
            radius=30.0,
            position=pygame.Vector2(cx, cy),
            velocity=pygame.Vector2(0.0, 0.0),
            color=(255, 200, 80),
        ))

        G = self.sim_config.gravitational_constant
        count = 10
        for _ in range(count):
            r = random.uniform(80.0, 420.0)
            theta = random.uniform(0.0, 2.0 * math.pi)

            px = cx + r * math.cos(theta)
            py = cy + r * math.sin(theta)

            # speed = math.sqrt(G * central_mass / r)
            speed = random.uniform(40, 90)
            vx = -speed * math.sin(theta)
            vy = speed * math.cos(theta)

            self.world.add(MassObject(
                config=self.sim_config,
                mass=random.uniform(1.0, 5.0),
                radius=5.0,
                position=pygame.Vector2(px, py),
                velocity=pygame.Vector2(vx, vy),
            ))

    def _resolve_merges(self) -> None:
        """Funde pares sobrepostos. O mais massivo absorve o outro."""
        bodies = self.world.entities
        n = len(bodies)
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
                if not a.overlaps(b):
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

        if self.input.was_pressed(pygame.K_f):
            self.force_vectors.toggle()

        if self.input.was_pressed(pygame.K_t):
            self.future_trajectory.toggle()

        if self.input.was_mouse_button_pressed(
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

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Executa um passo físico completo usando RK4."""

        bodies = self.world.entities
        start = perf_counter()
        accelerations = self.integrator.step(
            bodies,
            fixed_delta_time,
            self.gravity,
        )
        elapsed = (perf_counter() - start) * 1000
        self.integration_time_last_step = elapsed
        for body, acceleration in zip(bodies, accelerations):
            body.last_force.update(
                acceleration * body.mass
            )
        self._resolve_merges()
        self.world.purge_inactive()

    def render(self, renderer) -> None:
        """Desenha a `World` com o alpha de interpolação atual."""
        alpha = self.clock.interpolation_alpha

        if self.force_vectors.enabled:
            for body in self.world.visible_entities(
                    self.camera,
                    alpha,
            ):
                position = (
                    body.previous_position.lerp(
                        body.position,
                        alpha,
                    )
                    if alpha < 1.0
                    else body.position
                )

                self.force_vectors.draw(
                    renderer,
                    self.camera,
                    position,
                    body.last_force,
                    scale=1,
                )

        spend_time = 0
        if self.future_trajectory.enabled:
            spend_time = self.future_trajectory.draw(
                renderer,
                self.camera,
                self.world.entities,
                self.gravity,
                self.clock.fixed_timestep,
            )

        self.world.render(
            renderer,
            self.camera,
            alpha,
        )

        selected = self.inspector.selected
        reference = self.relation_inspector.reference

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

        renderer.draw_text(
            f"Pressione F para {force_action} os vetores de força",
            (5, self.config.height - 55),
            size=25,
            color=(255, 255, 255),
        )

        renderer.draw_text(
            f"Pressione T para {trajectory_action} a trajetória futura{': {:.2f}ms'.format(spend_time * 1000) if self.future_trajectory.enabled else ''}",
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