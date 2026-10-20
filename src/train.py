import argparse
import math
import random
from pathlib import Path
from typing import Optional

import torch
import pygame

from src.config import Configs, SimulationConfig
from src.engine.application import Application
from src.engine.world import World
from src.entities.massobject import MassObject
from src.entities.rocket import Rocket
from src.physics.collision import collides
from src.physics.composite_acceleration import CompositeAccelerationModel
from src.physics.gravity import NewtonianGravity
from src.physics.rk4 import RK4Integrator

from src.ai.neural_population import NeuralPopulation
from src.physics.isolated_host_gravity import IsolatedHostGravity


class Experimenter(Application):
    """@brief Simulação enxuta: um host + N foguetes, um por rede.

    @details Física isolada (`IsolatedHostGravity`): foguetes sentem
    apenas o host, sem interação entre si. Cada rede decide o próprio
    foguete a cada passo físico; recompensa segue DESIGN §6-7. Cada
    geração sorteia host, α e ângulo — o GA nunca vê o mesmo cenário
    duas vezes.
    """

    _INITIAL_ZOOM = 0.1

    ROCKET_MASS = 40.0
    ROCKET_BODY_LENGTH = 16.0
    ROCKET_BODY_WIDTH = 7.0
    ROCKET_NOSE_LENGTH = 9.0
    ROCKET_THRUST_ACCELERATION = 1000.0
    ROCKET_COLOR = (220, 220, 230)
    ROCKET_NOSE_COLOR = (240, 110, 80)

    _HOST_MASS_RANGE = (400_000.0, 700_000.0)
    _HOST_RADIUS_RANGE = (500.0, 540.0)
    _ALPHA_RANGE = (1.5, 5.0)

    _C_THRUST = 0.02
    _C_ROTATE = 0.003
    _C_TIME = 0.001
    _TERMINAL_SUCCESS = 100.0
    _TERMINAL_COLLISION = -100.0
    _TERMINAL_ESCAPE = -50.0
    _TERMINAL_REDOCKED = -5.0
    _TERMINAL_FUEL = -20.0
    _TERMINAL_TIMEOUT = -5.0
    _MAX_STEPS = 3600
    _STABLE_R_BAND = 0.12
    _STABLE_V_BAND = 0.20
    _ESCAPE_FACTOR = 8.0
    _POTENTIAL_CAP = 4.0

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        max_generations: int = 50,
        method: str = "sexual",
        elite_count: int = 1,
        mutation_rate: float = 0.05,
        mutation_std: float = 0.02,
        base_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        """@brief Configura o ambiente e os parâmetros do GA.

        @details Não popula o mundo — chame `experiment_single_host`
        antes de `run()`.

        @param config Janela e loop principal.
        @param sim_config Física (G, softening, etc.).
        @param max_generations Total de gerações do treino.
        @param method "sexual" (crossover) ou "asexual" (clone).
        @param elite_count Quantos melhores ficam intactos.
        @param mutation_rate Probabilidade de mutar cada peso.
        @param mutation_std Desvio-padrão do ruído gaussiano.
        @param base_seed Semente mestra (reprodutibilidade).
        @param output_dir Diretório do checkpoint do campeão.
        """
        super().__init__(config)
        self.sim_config = sim_config

        self.gravity = NewtonianGravity(
            gravitational_constant=sim_config.gravitational_constant,
            softening=sim_config.gravitational_softening,
        )
        self.integrator = RK4Integrator()

        self.camera.set_zoom(self._INITIAL_ZOOM)

        self.world: World = World()
        self.rocket_host: Optional[MassObject] = None

        self.rockets: list[Rocket] = []
        self.networks: list = []
        self.population: Optional[NeuralPopulation] = None

        self.rewards: list[float] = []
        self.alive: list[bool] = []
        self.steps: list[int] = []
        self.theta_accum: list[float] = []
        self.ever_undocked: list[bool] = []
        self.terminal_reasons: list[Optional[str]] = []
        self.prev_phi: list[float] = []
        self.prev_angle: list[float] = []
        self._r_ref: list[float] = []
        self._v_ref: list[float] = []

        self.generation: int = 0
        self.max_generations: int = max_generations
        self.training_done: bool = False
        self.history: list[list[float]] = []

        self.method = method
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.base_seed = base_seed
        self.output_dir = output_dir or Path("checkpoints")

        self._EPISODE_MAX_STEPS = self._MAX_STEPS
        self.episode_step: int = 0
        self._batch_obs: list[list[float]] = []

    def experiment_single_host(
        self,
        population: Optional[NeuralPopulation] = None,
        host_mass: Optional[float] = None,
        host_radius: Optional[float] = None,
        host_color: tuple[int, int, int] = (255, 200, 80),
        rocket_alpha: Optional[float] = None,
        docking_base_angle: Optional[float] = None,
        host_position: Optional[pygame.Vector2] = None,
        host_velocity: Optional[pygame.Vector2] = None,
    ) -> None:
        """@brief Monta o experimento: 1 host e N foguetes.

        @details Se `population` é None, cria 1 foguete sem rede (para
        inspeção). Caso contrário, cria `len(population)` foguetes em
        ângulos equidistantes, um por rede. Parâmetros `None` são
        sorteados — chamar sem argumentos gera um cenário totalmente
        novo, útil na geração 0.

        @param population Redes a controlar (ou None para 1 foguete).
        @param host_mass Massa do host; sorteada se None.
        @param host_radius Raio do host; sorteado se None.
        @param host_color Cor do host.
        @param rocket_alpha α do alvo orbital; sorteado se None.
        @param docking_base_angle Ângulo base de acoplagem; sorteado
            se None.
        @param host_position Posição do host; centro do mundo se None.
        @param host_velocity Velocidade do host; zero se None.
        """
        self.world.clear()
        self.rockets = []
        self.networks = []
        self.rocket_host = None
        self.population = population

        if host_mass is None:
            host_mass = random.uniform(*self._HOST_MASS_RANGE)
        if host_radius is None:
            host_radius = random.uniform(*self._HOST_RADIUS_RANGE)
        if rocket_alpha is None:
            rocket_alpha = random.uniform(*self._ALPHA_RANGE)

        cx = self.sim_config.world_width * 0.5
        cy = self.sim_config.world_height * 0.5
        if host_position is None:
            host_position = pygame.Vector2(cx, cy)
        if host_velocity is None:
            host_velocity = pygame.Vector2(0.0, 0.0)

        host = self._spawn_host(
            mass=host_mass, radius=host_radius,
            position=host_position, velocity=host_velocity,
            color=host_color,
        )
        self.rocket_host = host

        if docking_base_angle is None:
            docking_base_angle = random.uniform(0.0, 2.0 * math.pi)

        if population is None:
            self._spawn_rocket(host, docking_base_angle, rocket_alpha)
            self.networks = [None]
        else:
            n = len(population)
            for i in range(n):
                # Ângulos equidistantes: rede i controla o foguete i.
                angle = docking_base_angle + 2.0 * math.pi * i / n
                self._spawn_rocket(host, angle, rocket_alpha)
            self.networks = list(population)

        self.gravity = IsolatedHostGravity(
            gravitational_constant=self.sim_config.gravitational_constant,
            softening=self.sim_config.gravitational_softening,
            host_index=0,
        )
        self.camera.set_position(host.position)

        self._init_reward_tracking()

    def _spawn_host(
        self,
        mass: float,
        radius: float,
        position: pygame.Vector2,
        velocity: Optional[pygame.Vector2] = None,
        color: tuple[int, int, int] = (255, 200, 80),
    ) -> MassObject:
        """@brief Cria e adiciona o host ao mundo.

        @param mass Massa.
        @param radius Raio.
        @param position Posição inicial.
        @param velocity Velocidade inicial; zero se None.
        @param color Cor RGB.
        @return O host criado.
        """
        host = MassObject(
            config=self.sim_config,
            mass=mass,
            radius=radius,
            position=pygame.Vector2(position),
            velocity=pygame.Vector2(velocity) if velocity is not None
                     else pygame.Vector2(0.0, 0.0),
            color=color,
        )
        self.world.add(host)
        return host

    def _spawn_rocket(
        self,
        host: Optional[MassObject],
        angle: float,
        alpha: float,
    ) -> Rocket:
        """@brief Cria um foguete, opcionalmente acoplado a `host`.

        @param host Corpo ao qual acoplar; None deixa livre.
        @param angle Ângulo radial de acoplagem (ou orientação inicial).
        @param alpha α do alvo orbital do foguete.
        @return O foguete criado.
        """
        rocket = Rocket(
            config=self.sim_config,
            mass=self.ROCKET_MASS,
            position=pygame.Vector2(0.0, 0.0),
            velocity=pygame.Vector2(0.0, 0.0),
            alpha=alpha,
            angle=angle,
            body_length=self.ROCKET_BODY_LENGTH,
            body_width=self.ROCKET_BODY_WIDTH,
            nose_length=self.ROCKET_NOSE_LENGTH,
            thrust_acceleration=self.ROCKET_THRUST_ACCELERATION,
            color=self.ROCKET_COLOR,
            nose_color=self.ROCKET_NOSE_COLOR,
        )
        self.world.add(rocket)
        self.rockets.append(rocket)
        if host is not None:
            rocket.dock_to(host, angle)
        return rocket

    def _build_observation_for(
            self, rocket: Rocket, host: MassObject,
            out: Optional[list[float]] = None,
    ) -> Optional[list[float]]:
        """@brief Vetor de 9 floats do DESIGN §5 para (rocket, host).

        @details Escreve em `out` quando fornecido (reaproveitamento);
        caso contrário, aloca uma lista nova. Devolve None quando a
        observação é indefinida (corpos inativos, distância nula,
        parâmetros degenerados).

        @param rocket Foguete observado.
        @param host Corpo de referência.
        @param out Buffer reutilizável de tamanho ≥ 9; criado se None.
        @return O vetor preenchido, ou None.
        """
        if out is None:
            out = [0.0] * 9
        if rocket is None or host is None:
            return None
        if not rocket.active or not host.active:
            return None

        G = self.sim_config.gravitational_constant
        mu = G * (host.mass + rocket.mass)
        if mu <= 0.0:
            return None
        alpha = float(rocket.alpha)
        if alpha <= 0.0:
            return None
        r_ref = alpha * host.radius
        if r_ref <= 0.0:
            return None

        v_ref = math.sqrt(mu / r_ref)
        a_g_ref = mu / (r_ref * r_ref)
        omega_ref = v_ref / r_ref

        rp_x = rocket.position.x - host.position.x
        rp_y = rocket.position.y - host.position.y
        r = math.sqrt(rp_x * rp_x + rp_y * rp_y)
        if r <= 1e-12:
            return None
        rv_x = rocket.velocity.x - host.velocity.x
        rv_y = rocket.velocity.y - host.velocity.y
        v_r = (rp_x * rv_x + rp_y * rv_y) / r
        h = rp_x * rv_y - rp_y * rv_x
        v_t = h / r

        # cos/sin do ângulo bico-radial via projeções sobre r̂.
        nx = math.cos(rocket.angle)
        ny = math.sin(rocket.angle)
        cos_rel = (nx * rp_x + ny * rp_y) / r
        if cos_rel > 1.0:
            cos_rel = 1.0
        elif cos_rel < -1.0:
            cos_rel = -1.0
        sin_rel = (rp_x * ny - rp_y * nx) / r

        docked = 1.0 if rocket.docked_to is not None else 0.0
        fuel_frac = (
            rocket.fuel / rocket.fuel_max
            if rocket.fuel_max > 0.0 else 0.0
        )
        thrust_ratio = (
            rocket.thrust_acceleration / a_g_ref
            if a_g_ref > 0.0 else 1.0
        )
        rot_ratio = (
            rocket.rotation_speed / omega_ref
            if omega_ref > 0.0 else 1.0
        )
        log_thrust = max(-1.0, min(2.0, math.log10(max(thrust_ratio, 1e-6))))
        log_rot = max(-1.0, min(2.0, math.log10(max(rot_ratio, 1e-6))))

        out[0] = r / r_ref
        out[1] = v_r / v_ref
        out[2] = v_t / v_ref
        out[3] = sin_rel
        out[4] = cos_rel
        out[5] = docked
        out[6] = fuel_frac
        out[7] = log_thrust
        out[8] = log_rot
        return out

    def _rocket_kinematics(
            self, i: int
    ) -> tuple[float, float, float, float, float]:
        """@brief Cinemática relativa e potencial Φ de um foguete.

        @details Devolve `(r, v_r, v_t, angle, phi)` numa única
        passada usando só floats — sem `Vector2` temporário.

        @param i Índice em `self.rockets`.
        @return `(distância, v_radial, v_tangencial, ângulo, Φ)`.
        """
        rocket = self.rockets[i]
        host = self.rocket_host
        if host is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rp_x = rocket.position.x - host.position.x
        rp_y = rocket.position.y - host.position.y
        r = math.sqrt(rp_x * rp_x + rp_y * rp_y)
        if r <= 1e-12:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rx = rp_x / r
        ry = rp_y / r
        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        v_r = rvx * rx + rvy * ry
        # Momento angular h = r × v; v_t = h/r, assinada pelo sentido.
        h = rp_x * rvy - rp_y * rvx
        v_t = h / r
        angle = math.atan2(rp_y, rp_x)

        r_ref = self._r_ref[i]
        v_ref = self._v_ref[i]
        if r_ref > 0.0 and v_ref > 0.0:
            e_raio = min(abs(r / r_ref - 1.0), self._POTENTIAL_CAP)
            e_radial = min(abs(v_r) / v_ref, self._POTENTIAL_CAP)
            e_tang = min(abs(abs(v_t) - v_ref) / v_ref, self._POTENTIAL_CAP)
            phi = -(2.0 * e_raio + 0.75 * e_radial + 0.75 * e_tang)
        else:
            phi = 0.0

        return r, v_r, v_t, angle, phi

    def _update_rewards(self, host: MassObject) -> None:
        """@brief Atualiza a recompensa de todos os foguetes vivos.

        @details Roda após `try_land` (detecta re-acoplagem) e antes
        de `_resolve_merges` (detecta colisão antes da absorção).

        @param host Corpo hospedeiro atual.
        """
        for i in range(len(self.rockets)):
            if not self.alive[i]:
                continue
            rocket = self.rockets[i]
            if not rocket.active:
                # Absorvido por `_resolve_merges` em passo anterior.
                self.alive[i] = False
                continue
            self._update_reward_for(i, rocket, host)

    def _update_reward_for(
            self, i: int, rocket: Rocket, host: MassObject
    ) -> None:
        """@brief Recompensa e terminais de um foguete (DESIGN §6-7).

        @details Ordem dos terminais: sucesso → colisão → fuga →
        re-acoplagem → combustível → timeout.

        @param i Índice em `self.rockets`.
        @param rocket Foguete avaliado.
        @param host Corpo hospedeiro.
        """
        r, v_r, v_t, angle, new_phi = self._rocket_kinematics(i)
        r_ref = self._r_ref[i]
        v_ref = self._v_ref[i]

        # Shaping telescopa: ΣΦ(s') − ΣΦ(s) = Φ(final) − Φ(inicial).
        shaping = new_phi - self.prev_phi[i]
        self.prev_phi[i] = new_phi

        cost_thrust = self._C_THRUST if rocket.thrusting else 0.0
        cost_rotate = self._C_ROTATE * abs(rocket.rotation_input)
        self.rewards[i] += shaping - cost_thrust - cost_rotate - self._C_TIME
        self.steps[i] += 1

        in_band = (
                r_ref > 0.0 and v_ref > 0.0
                and abs(r / r_ref - 1.0) < self._STABLE_R_BAND
                and abs(v_r) / v_ref < self._STABLE_V_BAND
                and abs(abs(v_t) - v_ref) / v_ref < self._STABLE_V_BAND
        )

        # Progresso angular zera fora da faixa, impedindo "colar" na órbita.
        if in_band:
            delta = angle - self.prev_angle[i]
            # Normaliza Δθ para (-π, π] antes de acumular.
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            self.theta_accum[i] += delta
        else:
            self.theta_accum[i] = 0.0
        self.prev_angle[i] = angle

        # Só considera "re-acoplagem" quem já desacoplou ao menos 1×.
        if rocket.docked_to is None:
            self.ever_undocked[i] = True

        terminal: Optional[float] = None
        reason: Optional[str] = None

        if in_band and abs(self.theta_accum[i]) >= 2.0 * math.pi:
            terminal, reason = self._TERMINAL_SUCCESS, "success"
        elif (
                self.ever_undocked[i]
                and collides(host, rocket)
                and (host.velocity - rocket.velocity).length()
                >= self.sim_config.merge_min_relative_speed
        ):
            terminal, reason = self._TERMINAL_COLLISION, "collision"
        elif r > self._ESCAPE_FACTOR * host.radius:
            terminal, reason = self._TERMINAL_ESCAPE, "escape"
        elif self.ever_undocked[i] and rocket.docked_to is not None:
            terminal, reason = self._TERMINAL_REDOCKED, "redocked"
        elif rocket.fuel <= 0.0:
            terminal, reason = self._TERMINAL_FUEL, "fuel_exhausted"
        elif self.steps[i] >= self._MAX_STEPS:
            terminal, reason = self._TERMINAL_TIMEOUT, "timeout"

        if terminal is not None:
            self.rewards[i] += terminal
            self.alive[i] = False
            self.terminal_reasons[i] = reason

    def _init_reward_tracking(self) -> None:
        """@brief Zera buffers e pré-computa constantes por foguete.

        @details `r_ref` e `v_ref` são fixos durante o experimento
        (α e raio do host não mudam) — evita `sqrt` por passo.
        `prev_phi` recebe Φ(s₀) para que o shaping do primeiro passo
        seja exato.
        """
        n = len(self.rockets)
        self.episode_step = 0
        self.rewards = [0.0] * n
        self.alive = [True] * n
        self.steps = [0] * n
        self.theta_accum = [0.0] * n
        self.ever_undocked = [False] * n
        self.terminal_reasons = [None] * n
        self.prev_phi = [0.0] * n
        self.prev_angle = [0.0] * n
        self._r_ref = [0.0] * n
        self._v_ref = [1.0] * n
        self._batch_obs = [[0.0] * 9 for _ in range(n)]

        host = self.rocket_host
        if host is None:
            return

        G = self.sim_config.gravitational_constant
        for i, rocket in enumerate(self.rockets):
            r_ref = rocket.alpha * host.radius
            mu = G * (host.mass + rocket.mass)
            if r_ref > 0.0 and mu > 0.0:
                self._r_ref[i] = r_ref
                self._v_ref[i] = math.sqrt(mu / r_ref)

            _, _, _, angle, phi = self._rocket_kinematics(i)
            self.prev_phi[i] = phi
            self.prev_angle[i] = angle

    @staticmethod
    def load_network(path: Path):
        """@brief Carrega um `ActorCritic` de um checkpoint `.pt`.

        @details Aceita `state_dict` puro ou dicionário com chave
        `"model"`. Retorna em CPU e `eval()`, pronto para inferência.

        @param path Caminho do arquivo.
        @return Instância de `ActorCritic`.
        """
        import torch
        from src.ai.network import ActorCritic

        net = ActorCritic(obs_dim=9, n_actions=6, hidden_dim=128)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state)
        net.eval()
        return net

    def _resolve_merges(self) -> None:
        """@brief Funde o host com foguetes que colidem em alta velocidade.

        @details Sobrevive o mais massivo. `rocket_host` é atualizado
        se o host for absorvido; `purge_inactive` remove os absorvidos
        do mundo logo em seguida.
        """
        host = self.rocket_host
        if host is None or not host.active:
            return
        min_speed = self.sim_config.merge_min_relative_speed
        for rocket in self.rockets:
            if not rocket.active or rocket is host:
                continue
            if not collides(host, rocket):
                continue
            if (host.velocity - rocket.velocity).length() < min_speed:
                continue
            if host.mass >= rocket.mass:
                host.merge_with(rocket)
            else:
                rocket.merge_with(host)
                self.rocket_host = rocket
                break

    def get_fitness(self) -> list[float]:
        """@brief Recompensa acumulada de cada foguete/rede.

        @return Cópia de `self.rewards`, na ordem de `self.rockets`.
        """
        return list(self.rewards)

    def _end_generation(self) -> None:
        """@brief Encerra a geração: loga, evolui ou para o treino.

        @details Se `generation + 1 >= max_generations`, salva o
        campeão e chama `stop()`. Caso contrário, evolui a população
        e reconstrói o mundo in-place — a janela nunca fecha.
        """
        fitness = self.get_fitness()
        self.history.append(fitness)

        best = max(fitness)
        avg = sum(fitness) / len(fitness)
        counts: dict[str, int] = {}
        for reason in self.terminal_reasons:
            key = reason if reason is not None else "running"
            counts[key] = counts.get(key, 0) + 1

        print(
            f"[gen {self.generation:3d}] "
            f"best={best:+7.2f}  avg={avg:+7.2f}  terminais={counts}"
        )

        if self.generation + 1 >= self.max_generations:
            self.training_done = True
            self._save_champion(fitness)
            self.stop()
            return

        if self.population is None:
            self.training_done = True
            self.stop()
            return

        self.population = self.population.evolve(
            fitness,
            method=self.method,
            elite_count=self.elite_count,
            mutation_rate=self.mutation_rate,
            mutation_std=self.mutation_std,
            seed=self.base_seed + self.generation + 1,
        )
        self.generation += 1
        self.experiment_single_host(self.population)

    def _save_champion(self, fitness: list[float]) -> None:
        """@brief Salva em disco o campeão da última geração.

        @param fitness Lista de recompensas por indivíduo.
        """
        if self.population is None or not fitness:
            return
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"champion_gen_{self.generation:03d}.pt"
        torch.save(self.population[best_idx].state_dict(), path)
        print(
            f"[save] campeão da gen {self.generation} "
            f"(fitness {fitness[best_idx]:+.2f}) em {path}"
        )

    def update(self, delta_time: float) -> None:
        """@brief Zoom e pan da câmera (sem controle manual do foguete)."""
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(
                self.input.mouse_position,
                1.1 ** self.input.mouse_wheel,
            )

        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(
                -self.input.mouse_delta / self.camera.zoom
            )

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Passo físico: decisão, integração, recompensa, transição.

        @details A geração encerra em três casos: todos morreram, o
        limite global de passos foi atingido, ou o host ficou inativo
        (defensivo — impede loop infinito). Nesses casos, sobreviventes
        recebem timeout e `_end_generation` reconstrói o experimento.

        @param fixed_delta_time Passo fixo em segundos.
        """
        self.episode_step += 1
        host = self.rocket_host

        if host is not None and host.active and self.networks:
            for i, (rocket, network) in enumerate(
                    zip(self.rockets, self.networks)
            ):
                if (
                        network is None
                        or not rocket.active
                        or not self.alive[i]
                ):
                    continue
                obs = self._build_observation_for(rocket, host)
                if obs is None:
                    continue
                try:
                    rocket.control_with_network(
                        network, obs, deterministic=True,
                    )
                except Exception as exc:
                    print(f"[ai] falha ao consultar a rede ({exc})")

        bodies = self.world.entities
        self.world.fixed_update(fixed_delta_time)

        acceleration_model = CompositeAccelerationModel(
            self.gravity,
            bodies,
        )
        self.integrator.step(
            bodies,
            fixed_delta_time,
            acceleration_model,
        )

        for body in bodies:
            body.post_integrate()

        if host is not None and host.active:
            landing_targets = [host]
            for i, rocket in enumerate(self.rockets):
                if rocket.active and self.alive[i]:
                    rocket.try_land(
                        landing_targets,
                        self.sim_config.landing_max_speed,
                    )

        if host is not None and host.active:
            self._update_rewards(host)

        if (
                not any(self.alive)
                or self.episode_step >= self._EPISODE_MAX_STEPS
                or host is None
                or not host.active
        ):
            # Sobreviventes no limite global recebem timeout explícito.
            for i in range(len(self.alive)):
                if self.alive[i]:
                    self.rewards[i] += self._TERMINAL_TIMEOUT
                    self.alive[i] = False
                    if self.terminal_reasons[i] is None:
                        self.terminal_reasons[i] = "timeout"

            self._end_generation()
            return

        self._resolve_merges()
        self.world.purge_inactive()

    def render(self, renderer) -> None:
        """@brief Desenha o mundo e o HUD (geração e vivos).

        @param renderer Renderizador alvo.
        """
        alpha = self.clock.interpolation_alpha
        self.world.render(renderer, self.camera, alpha)

        n_alive = sum(self.alive) if self.alive else 0
        n_total = len(self.alive) if self.alive else 0

        renderer.draw_text(
            f"Geração {self.generation + 1}/{self.max_generations}",
            (10, 10),
            size=28,
            color=(255, 255, 255),
        )
        renderer.draw_text(
            f"Vivos: {n_alive}/{n_total}  |  passo: {self.episode_step}",
            (10, 42),
            size=22,
            color=(180, 220, 255),
        )


def main() -> None:
    """@brief Ponto de entrada: monta a geração 0 e roda o treino."""
    config = Configs(
        width=1500,
        height=1000,
        title="Experimenter",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    sim_config = SimulationConfig()

    population = NeuralPopulation(size=40, seed=42)

    app = Experimenter(
        config, sim_config,
        max_generations=50,
        method="sexual",
        elite_count=1,
        mutation_rate=0.05,
        mutation_std=0.02,
        base_seed=42,
    )
    app.experiment_single_host(population)
    app.run()


if __name__ == "__main__":
    main()