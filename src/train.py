import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
import pygame

from src.config import Configs, SimulationConfig
from src.engine.application import Application
from src.engine.world import World
from src.entities.massobject import MassObject
from src.entities.rocket import Rocket
from src.entities.entity_picker import EntityPicker
from src.physics.collision import collides
from src.physics.composite_acceleration import CompositeAccelerationModel
from src.physics.gravity import NewtonianGravity
from src.physics.rk4 import RK4Integrator

from src.ai.neural_population import NeuralPopulation
from src.physics.isolated_host_gravity import IsolatedHostGravity


class Experimenter(Application):
    """@brief Simulação enxuta: um host + N foguetes, um por rede.

    @details Física isolada (foguetes sentem apenas o host). Cada
    geração sorteia host, α e ângulo; a rede decide a ação de cada
    foguete a cada passo físico. Métricas são anexadas em
    `training_log.jsonl` ao fim da geração.
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
    _ALPHA_RANGE = (1.5, 3.0)

    # --- Custos densos e recompensas ---------------------------------
    _C_THRUST = 0.005                # Custo por passo com motor ativo.
    _C_ROTATE = 0.003                # Custo por passo com rotação ativa.
    _C_TIME = 0.0002                 # Custo por passo decorrido (tempo).
    _C_IN_BAND = 0.02                # Recompensa por passo em banda estável.
    _TERMINAL_SUCCESS = 100.0        # Bônus terminal ao completar 2π em banda.
    _BONUS_HALF_ORBIT = 50.0         # Bônus único ao cruzar π em banda (não-terminal).
    _TERMINAL_COLLISION = -100.0     # Penalidade terminal por colisão destrutiva.
    _TERMINAL_ESCAPE = -50.0         # Penalidade terminal ao ultrapassar 8·R_host.
    _TERMINAL_REDOCKED = -50.0        # Penalidade terminal por reacoplar após decolar.
    _TERMINAL_FUEL = -20.0           # Penalidade terminal por combustível esgotado.
    _TERMINAL_TIMEOUT = -30.0         # Penalidade terminal ao esgotar passos.

    # --- Geometria e física ------------------------------------------
    _TIMEOUT_ORBITS = 2.0            # Timeout = este nº × período orbital no r_ref.
    _STABLE_R_BAND = 0.05            # Tolerância relativa em r/r_ref para "em banda".
    _STABLE_V_BAND = 0.10            # Tolerância relativa em v_r/v_ref e v_t/v_ref.
    _ESCAPE_FACTOR = 8.0             # Múltiplo de R_host que caracteriza fuga.
    _POTENTIAL_CAP = 4.0             # Teto de cada termo de erro no Φ (anti-explosão).


    MUTATION_RATE_MAX = 0.06
    MUTATION_RATE_MIN = 0.02

    _ALPHA_MIN = 1.5
    _ALPHA_MAX_START = 2.5
    _ALPHA_MAX_END = 5.0

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        max_generations: int = 50,
        method: str = "sexual",
        elite_count: int = 1,
        mutation_std: float = 0.02,
        base_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        """@brief Configura o ambiente e o GA; não popula o mundo.

        @details Chame `experiment_single_host` antes de `run()`.

        @param config Janela e loop principal.
        @param sim_config Física (G, softening, etc.).
        @param max_generations Total de gerações do treino.
        @param method "sexual" (crossover) ou "asexual" (clone).
        @param elite_count Quantos melhores ficam intactos.
        @param mutation_rate Probabilidade de mutar cada peso.
        @param mutation_std Desvio-padrão do ruído gaussiano.
        @param base_seed Semente mestra.
        @param output_dir Diretório do checkpoint e do log.
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
        self._best_ever: float = -math.inf
        self._best_ever_gen: int = -1
        self._best_ever_state: Optional[dict] = None
        self._shutdown_done: bool = False

        self.method = method
        self.elite_count = elite_count
        self.mutation_std = mutation_std
        self.base_seed = base_seed
        self.output_dir = output_dir or Path("checkpoints")
        self.log_path: Path = self.output_dir / "training_log.jsonl"
        self._gen_start_time: float = 0.0

        self._current_alpha: float = 0.0
        self._EPISODE_MAX_STEPS: int = 0
        self.episode_step: int = 0
        self._batch_obs: list[list[float]] = []

        self.entity_picker = EntityPicker()
        self.selected_rocket: Optional[Rocket] = None
        self._selected_rocket_idx: Optional[int] = None
        self._selected_obs: list[float] = [0.0] * 9

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
        """@brief Monta o experimento: 1 host e N foguetes (1 por rede).

        @details Sem `population`, cria 1 foguete sem rede (inspeção).
        Com `population`, cria N foguetes em ângulos equidistantes. Os
        parâmetros `None` são sorteados — chamar sem argumentos gera um
        cenário novo.

        @param population Redes a controlar, ou None para 1 foguete.
        @param host_mass Massa do host; sorteada se None.
        @param host_radius Raio do host; sorteado se None.
        @param host_color Cor do host.
        @param rocket_alpha α do alvo orbital; sorteado se None.
        @param docking_base_angle Ângulo base de acoplagem; sorteado se None.
        @param host_position Posição do host; centro do mundo se None.
        @param host_velocity Velocidade do host; zero se None.
        """
        self.world.clear()
        self.rockets = []
        self.networks = []
        self.rocket_host = None
        self.population = population

        self.selected_rocket = None
        self._selected_rocket_idx = None

        if host_mass is None:
            host_mass = random.uniform(*self._HOST_MASS_RANGE)
        if host_radius is None:
            host_radius = random.uniform(*self._HOST_RADIUS_RANGE)
        if rocket_alpha is None:
            rocket_alpha = random.uniform(*self._current_alpha_range())

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
                angle = docking_base_angle + 2.0 * math.pi * i / n
                self._spawn_rocket(host, angle, rocket_alpha)
            self.networks = list(population)

        self._current_alpha = float(rocket_alpha)
        self.gravity = IsolatedHostGravity(
            gravitational_constant=self.sim_config.gravitational_constant,
            softening=self.sim_config.gravitational_softening,
            host_index=0,
        )
        self.camera.set_position(host.position)
        self._gen_start_time = perf_counter()
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
        @param alpha α do alvo orbital.
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
        """@brief Vetor de 9 floats do DESIGN §5.

        @details Escreve em `out` quando fornecido; senão aloca. Devolve
        None em estados degenerados (corpos inativos, r≈0, parâmetros
        inválidos).

        @param rocket Foguete observado.
        @param host Corpo de referência.
        @param out Buffer reutilizável de tamanho ≥ 9.
        @return Vetor preenchido, ou None.
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
        """@brief Cinemática relativa e potencial Φ do foguete `i`.

        @details Usa só floats; atualiza `self.max_r[i]` de passagem.

        @param i Índice em `self.rockets`.
        @return `(r, v_r, v_t, ângulo, Φ)`.
        """
        rocket = self.rockets[i]
        host = self.rocket_host
        if host is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rp_x = rocket.position.x - host.position.x
        rp_y = rocket.position.y - host.position.y
        r = math.sqrt(rp_x * rp_x + rp_y * rp_y)
        self.max_r[i] = max(self.max_r[i], r)
        if r <= 1e-12:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rx = rp_x / r
        ry = rp_y / r
        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        v_r = rvx * rx + rvy * ry
        h = rp_x * rvy - rp_y * rvx
        v_t = h / r
        angle = math.atan2(rp_y, rp_x)

        r_ref = self._r_ref[i]
        v_ref = self._v_ref[i]
        if r_ref > 0.0 and v_ref > 0.0:
            e_raio = min(abs(r / r_ref - 1.0), self._POTENTIAL_CAP)
            e_radial = min(abs(v_r) / v_ref, self._POTENTIAL_CAP)
            e_tang = min(abs(abs(v_t) - v_ref) / v_ref, self._POTENTIAL_CAP)
            phi = -5.0 * (2.0 * e_raio + 0.75 * e_radial + 0.75 * e_tang)
        else:
            phi = 0.0

        return r, v_r, v_t, angle, phi

    def _update_rewards(self, host: MassObject) -> None:
        """@brief Atualiza recompensa de todos os foguetes vivos.

        @details Roda após `try_land` e antes de `_resolve_merges`.
        Foguetes absorvidos por fusão são marcados mortos.

        @param host Corpo hospedeiro.
        """
        for i in range(len(self.rockets)):
            if not self.alive[i]:
                continue
            rocket = self.rockets[i]
            if not rocket.active:
                self.alive[i] = False
                continue
            self._update_reward_for(i, rocket, host)

    def _update_reward_for(
            self, i: int, rocket: Rocket, host: MassObject
    ) -> None:
        """@brief Recompensa e terminais do foguete `i`.

        @details Ordem dos terminais: sucesso → colisão → fuga →
        reacoplagem → combustível → timeout. Bônus de meia-órbita é
        não-terminal, disparado uma única vez.

        @param i Índice em `self.rockets`.
        @param rocket Foguete avaliado.
        @param host Corpo hospedeiro.
        """
        r, v_r, v_t, angle, new_phi = self._rocket_kinematics(i)
        r_ref = self._r_ref[i]
        v_ref = self._v_ref[i]

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

        if in_band:
            self.rewards[i] += self._C_IN_BAND
            self.in_band_steps[i] += 1
            if (
                    not self.half_orbit_awarded[i]
                    and abs(self.theta_accum[i]) >= math.pi
            ):
                self.half_orbit_awarded[i] = True
                self.rewards[i] += self._BONUS_HALF_ORBIT

        if in_band:
            delta = angle - self.prev_angle[i]
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            self.theta_accum[i] += delta
        else:
            self.theta_accum[i] = 0.0
        self.prev_angle[i] = angle

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
        elif self.ever_undocked[i] and rocket.docked_to is not None:
            if self.max_r[i] < 1.2 * host.radius:
                terminal, reason = self._TERMINAL_TIMEOUT, "never_launched"
            else:
                terminal, reason = self._TERMINAL_REDOCKED, "redocked"
        elif r > self._ESCAPE_FACTOR * host.radius:
            terminal, reason = self._TERMINAL_ESCAPE, "escape"
        elif rocket.fuel <= 0.0:
            terminal, reason = self._TERMINAL_FUEL, "fuel_exhausted"
        elif self.steps[i] >= self._EPISODE_MAX_STEPS:
            terminal, reason = self._TERMINAL_TIMEOUT, "timeout"

        if terminal is not None:
            self.rewards[i] += terminal
            self.alive[i] = False
            self.terminal_reasons[i] = reason

    def _init_reward_tracking(self) -> None:
        """@brief Zera buffers e pré-computa constantes por foguete.

        @details `r_ref`, `v_ref` e `_EPISODE_MAX_STEPS` são fixos na
        geração. `prev_phi` e `prev_angle` recebem o estado inicial
        para que o shaping do primeiro passo seja exato.
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
        self.max_r = [0.0] * n
        self.half_orbit_awarded = [False] * n
        self.in_band_steps = [0] * n

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

        if self._r_ref and self._r_ref[0] > 0.0:
            r_ref = self._r_ref[0]
            mu = G * (host.mass + self.rockets[0].mass)
            T_orbit = 2.0 * math.pi * math.sqrt(r_ref ** 3 / mu)
            self._EPISODE_MAX_STEPS = max(
                1, int(self._TIMEOUT_ORBITS * T_orbit / self.config.fixed_timestep)
            )
        else:
            self._EPISODE_MAX_STEPS = 100_000

    @staticmethod
    def load_network(path: Path):
        """@brief Carrega `ActorCritic` de um checkpoint `.pt`.

        @details Aceita `state_dict` puro ou dicionário com chave
        `"model"`. Devolve em CPU, `eval()`.

        @param path Caminho do arquivo.
        @return Instância de `ActorCritic`.
        """
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
        se o host for absorvido.
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
        """@brief Cópia da recompensa acumulada de cada foguete.

        @return Lista na ordem de `self.rockets`.
        """
        return list(self.rewards)

    def _write_generation_log(self, fitness: list[float]) -> None:
        """@brief Anexa uma linha JSONL com as métricas da geração.

        @details Uma linha por geração em `self.log_path`. Chamado
        antes de `evolve`, quando `self.rockets` ainda aponta para a
        geração concluída. Formato estável — alterar campos quebra
        análise histórica.

        @param fitness Recompensa final de cada indivíduo.
        """
        host = self.rocket_host
        if host is None or not fitness:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n = len(fitness)
        sorted_f = sorted(fitness)
        mean_f = sum(fitness) / n
        var_f = sum((x - mean_f) ** 2 for x in fitness) / n
        median_f = (
            sorted_f[n // 2] if n % 2 == 1
            else 0.5 * (sorted_f[n // 2 - 1] + sorted_f[n // 2])
        )

        r_ref = self._r_ref[0] if self._r_ref else 0.0
        v_ref = self._v_ref[0] if self._v_ref else 0.0
        mu = self.sim_config.gravitational_constant * (
            host.mass + (self.rockets[0].mass if self.rockets else 0.0)
        )
        T_orbit = (
            2.0 * math.pi * math.sqrt(r_ref ** 3 / mu)
            if r_ref > 0.0 and mu > 0.0 else 0.0
        )

        terminals: dict[str, int] = {}
        for reason in self.terminal_reasons:
            key = reason if reason is not None else "running"
            terminals[key] = terminals.get(key, 0) + 1

        individuals = []
        for i in range(n):
            rocket = self.rockets[i]
            steps_i = max(self.steps[i], 1)
            individuals.append({
                "idx": i,
                "reward": round(float(fitness[i]), 4),
                "terminal": self.terminal_reasons[i],
                "steps": int(self.steps[i]),
                "max_r_ratio": (
                    round(float(self.max_r[i] / r_ref), 4)
                    if r_ref > 0.0 else 0.0
                ),
                "in_band_ratio": round(
                    float(self.in_band_steps[i] / steps_i), 4
                ),
                "half_awarded": bool(self.half_orbit_awarded[i]),
                "fuel_used": round(float(rocket.fuel_max - rocket.fuel), 2),
            })

        best_idx = max(range(n), key=lambda i: fitness[i])

        entry = {
            "event": "generation",
            "gen": int(self.generation),
            "wall_time_s": round(perf_counter() - self._gen_start_time, 3),
            "alpha": round(float(self._current_alpha), 4),
            "host_mass": round(float(host.mass), 2),
            "host_radius": round(float(host.radius), 2),
            "r_ref": round(float(r_ref), 2),
            "v_ref": round(float(v_ref), 4),
            "orbital_period_s": round(float(T_orbit), 3),
            "max_steps": int(self._EPISODE_MAX_STEPS),
            "steps_run": int(self.episode_step),
            "population_size": n,
            "fitness": {
                "best": round(float(sorted_f[-1]), 4),
                "avg": round(float(mean_f), 4),
                "median": round(float(median_f), 4),
                "std": round(float(math.sqrt(var_f)), 4),
                "min": round(float(sorted_f[0]), 4),
            },
            "terminals": terminals,
            "best_idx": int(best_idx),
            "best_ever": (
                round(float(self._best_ever), 4)
                if self._best_ever != -math.inf else None
            ),
            "best_ever_gen": (
                int(self._best_ever_gen)
                if self._best_ever_gen >= 0 else None
            ),
            "individuals": individuals,
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _end_generation(self) -> None:
        """@brief Loga, evolui (ou encerra) e reconstrói o experimento.

        @details Idempotente: chamadas após `training_done=True` são
        no-op. No caminho de "max_generations atingido", NÃO salva a
        rede aqui — `shutdown()` faz isso, cobrindo também o caso de
        janela fechada antes do fim.
        """
        if self.training_done:
            return

        fitness = self.get_fitness()
        self.history.append(fitness)

        best = max(fitness)
        avg = sum(fitness) / len(fitness)
        gen_dt = perf_counter() - self._gen_start_time

        if best > self._best_ever:
            self._best_ever = best
            self._best_ever_gen = self.generation
            if self.population is not None:
                best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
                # Clone independente: o dict sobrevive à próxima evolução.
                self._best_ever_state = {
                    k: v.detach().clone()
                    for k, v in self.population[best_idx].state_dict().items()
                }

        counts: dict[str, int] = {}
        for reason in self.terminal_reasons:
            key = reason if reason is not None else "running"
            counts[key] = counts.get(key, 0) + 1

        self._write_generation_log(fitness)

        print(
            f"[gen {self.generation:3d}] α={self._current_alpha:.2f}  "
            f"best={best:+7.2f}  avg={avg:+7.2f}  "
            f"mut={self.current_mutation_rate():.3f}  "
            f"dt={gen_dt:.1f}s  terminais={counts}"
        )

        if self.generation + 1 >= self.max_generations:
            self.training_done = True
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
            mutation_rate=self.current_mutation_rate(),
            mutation_std=self.mutation_std,
            seed=self.base_seed + self.generation + 1,
        )
        self.generation += 1
        self.experiment_single_host(self.population)

    def current_mutation_rate(self):
        total = max(1, self.max_generations - 1)
        progress = min(1.0, self.generation / total)
        return (
            self.MUTATION_RATE_MAX * (1 - progress) + self.MUTATION_RATE_MIN * progress
        )

    def _current_alpha_range(self) -> tuple[float, float]:
        """@brief Intervalo efetivo de α para a geração atual.

        @details Interpolação linear entre `_ALPHA_MAX_START` e
        `_ALPHA_MAX_END` com o progresso do treino. Não altera
        `_ALPHA_RANGE` original (que pode ser usado em avaliação).
        """
        progress = min(
            1.0,
            self.generation / max(1, self.max_generations - 1),
        )
        hi = (
            self._ALPHA_MAX_START
            + (self._ALPHA_MAX_END - self._ALPHA_MAX_START) * progress
        )
        return (self._ALPHA_MIN, hi)

    def _save_best_ever(self) -> None:
        """@brief Salva a melhor rede global em `best_ever.pt`.

        @details Formato é `state_dict` puro (mesmo consumido por
        `load_network`). Arquivo é sobrescrito se já existir.
        """
        if self._best_ever_state is None:
            print("[shutdown] nenhuma geração concluída — sem rede a salvar.")
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "best_ever.pt"
            torch.save(self._best_ever_state, path)
            print(
                f"[save] melhor rede global: fitness {self._best_ever:+.2f} "
                f"(gen {self._best_ever_gen}) em {path}"
            )
        except Exception as exc:
            print(f"[shutdown] falha ao salvar melhor rede: {exc}")

    def _write_session_end_log(self) -> None:
        """@brief Anexa a entrada final da sessão ao JSONL.

        @details Última linha tem `event="session_end"`, permitindo
        distinguir de entradas de geração. Inclui motivo do
        encerramento e sumário do campeão.
        """
        if not self.log_path.parent.exists():
            return

        reason = "max_generations" if self.training_done else "interrupted"
        gens_completed = self.generation + (1 if self.training_done else 0)

        entry = {
            "event": "session_end",
            "reason": reason,
            "generations_completed": int(gens_completed),
            "best_ever": (
                round(float(self._best_ever), 4)
                if self._best_ever != -math.inf else None
            ),
            "best_ever_gen": (
                int(self._best_ever_gen)
                if self._best_ever_gen >= 0 else None
            ),
            "has_saved_network": self._best_ever_state is not None,
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[shutdown] falha ao escrever session_end: {exc}")

    def update(self, delta_time: float) -> None:
        """@brief Zoom, pan e seleção de foguete por clique.

        @param delta_time Tempo desde o último frame.
        """
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(
                self.input.mouse_position,
                1.1 ** self.input.mouse_wheel,
            )

        if self.input.was_pressed(pygame.K_RIGHT):
            self.speed_multiplier = min(3.0, self.speed_multiplier + 0.5)
        if self.input.was_pressed(pygame.K_LEFT):
            self.speed_multiplier = max(1.0, self.speed_multiplier - 0.5)

        if self.input.was_mouse_button_clicked(pygame.BUTTON_LEFT):
            pick_alpha = min(1.0, self.clock.interpolation_alpha)
            picked = self.entity_picker.pick(
                self.input.mouse_position,
                self.camera,
                self.world.entities,
                alpha=pick_alpha,
            )
            if isinstance(picked, Rocket):
                self.selected_rocket = picked
                try:
                    self._selected_rocket_idx = self.rockets.index(picked)
                except ValueError:
                    self.selected_rocket = None
                    self._selected_rocket_idx = None
            else:
                self.selected_rocket = None
                self._selected_rocket_idx = None

        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(
                -self.input.mouse_delta / self.camera.zoom
            )

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Passo físico: decisão, integração, recompensa, transição.

        @details A geração encerra quando todos morreram, o limite de
        passos é atingido, ou o host fica inativo. Sobreviventes nesse
        ponto recebem timeout explícito.

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

    def _draw_target_orbit(self, renderer) -> None:
        """@brief Desenha o círculo-alvo da geração e a banda de tolerância.

        @details Linha principal em `r_ref = α · host.radius`; duas
        linhas internas marcam `r_ref · (1 ± _STABLE_R_BAND)`.

        @param renderer Renderizador alvo.
        """
        host = self.rocket_host
        if host is None or not host.active:
            return
        if self._current_alpha <= 0.0:
            return

        center_screen = self.camera.world_to_screen(host.position)
        radius_screen = self.camera.world_to_screen_size(
            self._current_alpha * host.radius
        )

        renderer.draw_circle(
            (120, 200, 255), center_screen, radius_screen, width=1,
        )
        for factor in (
            1.0 - self._STABLE_R_BAND,
            1.0 + self._STABLE_R_BAND,
        ):
            renderer.draw_circle(
                (80, 140, 190),
                center_screen,
                self.camera.world_to_screen_size(
                    self._current_alpha * host.radius * factor
                ),
                width=1,
            )

    def _draw_observation_labels(
        self,
        renderer,
        rocket: Rocket,
        obs: list[float],
        idx: int,
    ) -> None:
        """@brief Desenha o vetor de observação no canto superior direito.

        @details `anchor="topright"` alinha a borda direita de cada
        linha sem medir texto. Sem alocação por frame: valores vêm de
        `self._selected_obs`.

        @param renderer Renderizador alvo.
        @param rocket Foguete selecionado.
        @param obs Buffer com os 9 valores.
        @param idx Índice do foguete em `self.rockets`.
        """
        x = self.config.width - 10
        y = 10
        line_height = 22

        renderer.draw_text(
            f"Foguete #{idx}",
            (x, y),
            size=24,
            color=(255, 255, 255),
            anchor="topright",
        )
        y += 30

        labels = (
            ("ALPHA", f"{rocket.alpha:.3f}"),
            ("r/r_ref", f"{obs[0]:+.3f}"),
            ("v_r/v_ref", f"{obs[1]:+.3f}"),
            ("v_t/v_ref", f"{obs[2]:+.3f}"),
            ("sin(θ_rel)", f"{obs[3]:+.3f}"),
            ("cos(θ_rel)", f"{obs[4]:+.3f}"),
            ("docked", "sim" if obs[5] > 0.5 else "não"),
            ("fuel/fuel_max", f"{obs[6]:.3f}"),
            ("log10(a_t/a_g)", f"{obs[7]:+.3f}"),
            ("log10(ω/ω_ref)", f"{obs[8]:+.3f}"),
        )
        for label, value in labels:
            renderer.draw_text(
                f"{label}: {value}",
                (x, y),
                size=20,
                color=(220, 240, 255),
                anchor="topright",
            )
            y += line_height

    def render(self, renderer) -> None:
        """@brief Desenha órbita-alvo, mundo, seleção e HUD.

        @param renderer Renderizador alvo.
        """
        alpha = self.clock.interpolation_alpha

        self._draw_target_orbit(renderer)

        if self.selected_rocket is not None and self.selected_rocket.active:
            self.selected_rocket.render_selection(
                renderer, self.camera, alpha,
            )

        speed_label = f"{self.speed_multiplier:.2f}x"
        renderer.draw_text(
            f"Velocidade = {speed_label}",
            (10, 134),
            size=22,
            color=(140, 220, 140),
        )

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

        host = self.rocket_host
        if host is not None and self._current_alpha > 0.0:
            renderer.draw_text(
                f"α = {self._current_alpha:.3f}",
                (10, 74),
                size=22,
                color=(255, 220, 120),
            )

        if self._EPISODE_MAX_STEPS > 0:
            renderer.draw_text(
                f"Max passos = {self._EPISODE_MAX_STEPS}",
                (10, 104),
                size=22,
                color=(255, 220, 120),
            )

        rocket = self.selected_rocket
        if rocket is None or not rocket.active:
            if rocket is not None:
                self.selected_rocket = None
                self._selected_rocket_idx = None
            return

        if host is None or not host.active:
            return

        idx = self._selected_rocket_idx
        if idx is None:
            return

        obs = self._build_observation_for(rocket, host, out=self._selected_obs)
        if obs is None:
            return

        self._draw_observation_labels(renderer, rocket, obs, idx)

    def shutdown(self) -> None:
        """@brief Finaliza a sessão: salva melhor rede e fecha o log.

        @details Chamado por `Application.run()` no `finally`, cobrindo
        os três caminhos de término: janela fechada (`quit_requested`),
        `max_generations` atingido, ou exceção no loop principal.
        Idempotente via `_shutdown_done`.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self._save_best_ever()
        self._write_session_end_log()
        super().shutdown()


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

    population = NeuralPopulation(size=100, seed=42)

    app = Experimenter(
        config, sim_config,
        max_generations=150,
        method="sexual",
        elite_count=5,
        mutation_std=0.03,
        base_seed=42,
    )
    app.experiment_single_host(population)
    print(f"[log] métricas em {app.log_path} (JSON Lines)")
    app.run()
    print(f"[done] log final: {app.log_path}")


if __name__ == "__main__":
    main()