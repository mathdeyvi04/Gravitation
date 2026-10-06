import math
import random
import pygame
from pathlib import Path
from src.config import Configs, SimulationConfig
from src.engine.application import Application
from src.engine.vector_visualizer import VectorVisualizer
from src.engine.trajectory_visualizer import TrajectoryVisualizer
from src.engine.world import World
from src.entities.massobject import MassObject


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

        # A câmera inicia no centro do mundo, que é o ponto para onde
        # `MassObject` distribui os corpos aleatórios. Sem isso, os
        # corpos nascem em `[0, W] × [0, H]` enquanto a câmera olha
        # para `[-W/2, +W/2] × [-H/2, +H/2]`, e quase tudo fica fora
        # da tela.
        self.camera.set_position(
            (sim_config.world_width * 0.5, sim_config.world_height * 0.5)
        )

        self.force_vectors = VectorVisualizer(
            color=(80, 220, 255)
        )
        self.future_trajectory = TrajectoryVisualizer(
            steps=sim_config.future_trajectory_steps,
            color=(170, 170, 255),
        )
        self._seed_solar_system()

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

        central_mass = 500.0
        self.world.add(MassObject(
            config=self.sim_config,
            mass=central_mass,
            radius=30.0,
            position=pygame.Vector2(cx, cy),
            velocity=pygame.Vector2(0.0, 0.0),
            color=(255, 200, 80),
        ))

        G = self.sim_config.gravitational_constant
        count = 4
        for _ in range(count):
            r = random.uniform(80.0, 420.0)
            theta = random.uniform(0.0, 2.0 * math.pi)

            px = cx + r * math.cos(theta)
            py = cy + r * math.sin(theta)

            # speed = math.sqrt(G * central_mass / r)
            speed = random.uniform(30, 80)
            vx = -speed * math.sin(theta)
            vy = speed * math.cos(theta)

            self.world.add(MassObject(
                config=self.sim_config,
                mass=random.uniform(1.0, 5.0),
                radius=5.0,
                position=pygame.Vector2(px, py),
                velocity=pygame.Vector2(vx, vy),
            ))

    def _apply_gravity(self) -> None:
        """Acumula a atração gravitacional em todos os pares (O(N²))."""
        bodies = self.world.entities
        n = len(bodies)
        G = self.sim_config.gravitational_constant
        eps = self.sim_config.gravitational_softening
        for i in range(n):
            a = bodies[i]
            for j in range(i + 1, n):
                a.apply_mutual_gravity(bodies[j], G, eps)

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

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Executa um passo físico completo."""
        self._apply_gravity()
        self.world.fixed_update(fixed_delta_time)
        self._resolve_merges()
        self.world.purge_inactive()

    def render(self, renderer) -> None:
        """Desenha a `World` com o alpha de interpolação atual."""
        alpha = self.clock.interpolation_alpha

        if self.force_vectors.enabled:
            for body in self.world.entities:
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
                    scale=1,
                )

        if self.future_trajectory.enabled:
            self.future_trajectory.draw(
                renderer,
                self.camera,
                self.world.entities,
                self.sim_config.gravitational_constant,
                self.sim_config.gravitational_softening,
                self.clock.fixed_timestep,
            )

        self.world.render(
            renderer,
            self.camera,
            alpha,
        )

        # ------------------------------------------------------
        # Apresentação de HUD

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
            f"Pressione T para {trajectory_action} a trajetória futura",
            (5, self.config.height - 25),
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