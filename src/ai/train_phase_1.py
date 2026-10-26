"""Fase 1 — Desacoplagem: as redes aprendem a sair da superfície.

O foguete começa acoplado. Cada geração roda `K` episódios por rede
(host e ângulos sorteados por episódio) e evolui a população por GA
com seleção por torneio.

Meta da fase: atingir `r ≥ 2·R_host` com a velocidade alinhada à
tangente do raio-alvo (dentro de ~53°) **e** velocidade tangencial
comparável à circular (`|v_t| ≥ 0.5·v_ref`). Bônus adicionais premiam
eficiência: menos combustível, menos tempo, melhor alinhamento.
"""

import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Optional

import pygame
import torch

from src.config import Configs, SimulationConfig
from src.engine.application import Application
from src.engine.world import World
from src.entities.massobject import MassObject
from src.entities.rocket import Rocket
from src.physics.collision import collides
from src.physics.composite_acceleration import CompositeAccelerationModel
from src.physics.isolated_host_gravity import IsolatedHostGravity
from src.physics.rk4 import RK4Integrator

from src.ai.neural_population import NeuralPopulation


class UndockingTrainer(Application):
    """@brief Fase 1 do currículo: ensinar redes a desacoplar.

    @details Estado inicial sempre acoplado. Cada geração roda `K`
    episódios (hosts sorteados) para reduzir o ruído do fitness; a
    evolução usa torneio para manter diversidade gênica. A função de
    recompensa premia alinhamento com a tangente e eficiência.
    """

    _INITIAL_ZOOM = 0.08

    # --- Foguete (fixo para todas as fases)
    ROCKET_MASS = 40.0
    ROCKET_BODY_LENGTH = 16.0
    ROCKET_BODY_WIDTH = 7.0
    ROCKET_NOSE_LENGTH = 9.0
    ROCKET_THRUST_ACCELERATION = 1000.0
    ROCKET_COLOR = (220, 220, 230)
    ROCKET_NOSE_COLOR = (240, 110, 80)

    # --- Host (sorteado por episódio)
    _HOST_MASS_RANGE = (400_000.0, 700_000.0)
    _HOST_RADIUS_RANGE = (500.0, 540.0)

    # --- α fixo na Fase 1 (só normaliza a observação)
    _ALPHA = 2.0

    # --- Metas / limites
    _SUCCESS_R_FACTOR = 2.0        # sucesso dispara em r ≥ 2·R_host
    _ESCAPE_R_FACTOR = 8.0         # fuga: r > 8·R_host

    # --- Shaping (Φ)
    # Altitude: 0 na superfície, 1 em r_ref. Tangencial: |v_t|/v_ref,
    # capado em 1. Somados, dão o gradiente que o GA precisa para
    # distinguir "subiu reto" de "subiu virando".
    _PHI_ALT_SCALE = 20.0
    _PHI_TAN_SCALE = 10.0

    # --- Custos densos (eficiência importa)
    _C_THRUST = 0.02               # ↑ de 0.005 (DESIGN §6 original)
    _C_ROTATE = 0.003
    _C_TIME = 0.001                # ↑ de 0.0002 (DESIGN §6 original)

    # --- Terminais
    _TERMINAL_SUCCESS = +100.0
    _TERMINAL_CRASH = -50.0
    _TERMINAL_REDOCKED = -50.0
    _TERMINAL_ESCAPE = -50.0
    _TERMINAL_FUEL = -30.0
    _TERMINAL_TIMEOUT = -50.0

    # --- Bônus no sucesso (qualidade)
    _SUCCESS_ALIGNMENT_MIN = 0.5   # |v_t|/|v| ≥ 0.5 (≤ ~53° da tangente)
    _SUCCESS_VT_MIN = 0.5          # |v_t|/v_ref ≥ 0.5 (não deixa ganhar
                                   # o bônus "no ápice com v_t ≈ 0")
    _ALIGNMENT_BONUS_SCALE = 50.0  # + alignment·scale em [0.5,1]
    _FUEL_BONUS_SCALE = 40.0       # + (fuel/fuel_max)·scale em [0,1]
    _TIME_BONUS_SCALE = 30.0       # + (1 - steps/max)·scale em [0,1]

    # --- Timeout: fração do período orbital na superfície
    _TIMEOUT_ORBITS = 0.75         # ↑ de 0.5 — dar fôlego ao gravity turn

    # --- Currículo
    _TOURNAMENT_SIZE = 3

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        max_generations: int = 50,             # ↓ de 100
        episodes_per_gen: int = 3,
        method: str = "sexual",
        elite_count: int = 5,
        mutation_rate_start: float = 0.05,
        mutation_rate_end: float = 0.01,
        mutation_std: float = 0.03,
        base_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(config)
        self.sim_config = sim_config
        self.integrator = RK4Integrator()
        self.camera.set_zoom(self._INITIAL_ZOOM)

        self.world: World = World()
        self.host: Optional[MassObject] = None
        self.rockets: list[Rocket] = []
        self.networks: list = []
        self.population: Optional[NeuralPopulation] = None

        # --- Estado por episódio
        self.rewards: list[float] = []
        self.alive: list[bool] = []
        self.steps: list[int] = []
        self.ever_undocked: list[bool] = []
        self.terminal_reasons: list[Optional[str]] = []
        self.prev_phi: list[float] = []
        self._v_ref: list[float] = []
        self._episode_step = 0
        self._episode_max_steps = 0
        self._episode_success_alignments: list[float] = []
        self._episode_success_fuel_fracs: list[float] = []
        self._episode_success_time_fracs: list[float] = []

        # --- Estado por geração
        self._gen_cumulative: list[float] = []
        self._gen_terminal_counts: dict[str, int] = {}
        self._gen_success_alignments: list[float] = []
        self._gen_success_fuel_fracs: list[float] = []
        self._gen_success_time_fracs: list[float] = []
        self._gen_episodes_done = 0
        self._gen_start_time = 0.0

        # --- Sessão
        self.generation = 0
        self.max_generations = max_generations
        self.training_done = False
        self._best_ever = -math.inf
        self._best_ever_gen = -1
        self._best_ever_state: Optional[dict] = None
        self._shutdown_done = False

        # --- Config do GA
        self.episodes_per_gen = episodes_per_gen
        self.method = method
        self.elite_count = elite_count
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_end = mutation_rate_end
        self.mutation_std = mutation_std
        self.base_seed = base_seed

        # --- Persistência
        self.output_dir = output_dir or Path("checkpoints")
        self.log_path = self.output_dir / "training_log.jsonl"

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def start_training(self, population: NeuralPopulation) -> None:
        """@brief Conecta a população inicial e inicia a geração 0."""
        self.population = population
        self.networks = list(population)
        self._gen_cumulative = [0.0] * len(population)
        self._gen_terminal_counts = {}
        self._gen_success_alignments = []
        self._gen_success_fuel_fracs = []
        self._gen_success_time_fracs = []
        self._gen_episodes_done = 0
        self._gen_start_time = perf_counter()
        self._start_episode()

    # ------------------------------------------------------------------
    # Ciclo de episódio / geração
    # ------------------------------------------------------------------

    def _start_episode(self) -> None:
        """@brief Limpa o mundo e monta um cenário de desacoplagem."""
        random.seed(
            self.base_seed
            + self.generation * self.episodes_per_gen
            + self._gen_episodes_done
        )

        self.world.clear()
        self.rockets = []

        host_mass = random.uniform(*self._HOST_MASS_RANGE)
        host_radius = random.uniform(*self._HOST_RADIUS_RANGE)
        cx = self.sim_config.world_width * 0.5
        cy = self.sim_config.world_height * 0.5

        self.host = self._spawn_host(
            mass=host_mass, radius=host_radius,
            position=pygame.Vector2(cx, cy),
        )

        n = len(self.networks)
        base_angle = random.uniform(0.0, 2.0 * math.pi)
        for i in range(n):
            angle = base_angle + 2.0 * math.pi * i / n
            self._spawn_rocket_docked(self.host, angle)

        self.gravity = IsolatedHostGravity(
            gravitational_constant=self.sim_config.gravitational_constant,
            softening=self.sim_config.gravitational_softening,
            host_index=0,
        )

        self.camera.set_position(self.host.position)
        self._init_episode_tracking()

    def _init_episode_tracking(self) -> None:
        n = len(self.rockets)
        self._episode_step = 0
        self.rewards = [0.0] * n
        self.alive = [True] * n
        self.steps = [0] * n
        self.ever_undocked = [False] * n
        self.terminal_reasons = [None] * n
        self.prev_phi = [0.0] * n
        self._episode_success_alignments = []
        self._episode_success_fuel_fracs = []
        self._episode_success_time_fracs = []

        host = self.host
        if host is not None and host.mass > 0.0 and host.radius > 0.0:
            mu = self.sim_config.gravitational_constant * (
                host.mass + self.ROCKET_MASS
            )
            r_ref = self._SUCCESS_R_FACTOR * host.radius
            v_ref = math.sqrt(mu / r_ref) if r_ref > 0.0 else 0.0
            self._v_ref = [v_ref] * n

            T_surface = 2.0 * math.pi * math.sqrt(host.radius ** 3 / mu)
            self._episode_max_steps = max(
                1,
                int(self._TIMEOUT_ORBITS * T_surface / self.config.fixed_timestep),
            )
        else:
            self._v_ref = [0.0] * n
            self._episode_max_steps = 100_000

        for i in range(n):
            self.prev_phi[i] = self._compute_phi(i)

    def _end_episode(self) -> None:
        """@brief Acumula fitness do episódio; reinicia ou evolui."""
        for i in range(len(self.rockets)):
            self._gen_cumulative[i] += self.rewards[i]
            reason = self.terminal_reasons[i] or "timeout"
            self._gen_terminal_counts[reason] = (
                self._gen_terminal_counts.get(reason, 0) + 1
            )

        self._gen_success_alignments.extend(self._episode_success_alignments)
        self._gen_success_fuel_fracs.extend(self._episode_success_fuel_fracs)
        self._gen_success_time_fracs.extend(self._episode_success_time_fracs)

        self._gen_episodes_done += 1

        if self._gen_episodes_done >= self.episodes_per_gen:
            self._end_generation()
        else:
            self._start_episode()

    def _end_generation(self) -> None:
        if self.training_done:
            return

        n = len(self._gen_cumulative)
        fitness = [r / self.episodes_per_gen for r in self._gen_cumulative]
        gen_dt = perf_counter() - self._gen_start_time

        if fitness and self.population is not None:
            best = max(fitness)
            if best > self._best_ever:
                self._best_ever = best
                self._best_ever_gen = self.generation
                best_idx = max(range(n), key=lambda i: fitness[i])
                self._best_ever_state = {
                    k: v.detach().clone()
                    for k, v in self.population[best_idx].state_dict().items()
                }

        self._write_generation_log(fitness, gen_dt)

        total = n * self.episodes_per_gen
        success = self._gen_terminal_counts.get("success", 0)
        success_rate = (success / total) if total > 0 else 0.0
        avg = (sum(fitness) / n) if n > 0 else 0.0
        best = max(fitness) if fitness else 0.0

        mean_align = (
            sum(self._gen_success_alignments) / len(self._gen_success_alignments)
            if self._gen_success_alignments else 0.0
        )
        mean_fuel = (
            sum(self._gen_success_fuel_fracs) / len(self._gen_success_fuel_fracs)
            if self._gen_success_fuel_fracs else 0.0
        )

        print(
            f"[gen {self.generation:3d}] "
            f"best={best:+7.2f}  avg={avg:+7.2f}  "
            f"succ={success_rate:.0%}  "
            f"align={mean_align:.2f}  fuel={mean_fuel:.2f}  "
            f"mut={self._current_mutation_rate():.3f}  "
            f"dt={gen_dt:.1f}s  "
            f"term={self._gen_terminal_counts}"
        )

        if self.generation + 1 >= self.max_generations or self.population is None:
            self.training_done = True
            self.stop()
            return

        self.population = self.population.evolve(
            fitness,
            method=self.method,
            elite_count=self.elite_count,
            mutation_rate=self._current_mutation_rate(),
            mutation_std=self.mutation_std,
            seed=self.base_seed + self.generation + 1,
            tournament_size=self._TOURNAMENT_SIZE,
        )
        self.networks = list(self.population)

        self.generation += 1
        self._gen_cumulative = [0.0] * len(self.networks)
        self._gen_terminal_counts = {}
        self._gen_success_alignments = []
        self._gen_success_fuel_fracs = []
        self._gen_success_time_fracs = []
        self._gen_episodes_done = 0
        self._gen_start_time = perf_counter()
        self._start_episode()

    def _current_mutation_rate(self) -> float:
        total = max(1, self.max_generations - 1)
        p = min(1.0, self.generation / total)
        return self.mutation_rate_start * (1.0 - p) + self.mutation_rate_end * p

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def _spawn_host(self, mass: float, radius: float,
                    position: pygame.Vector2) -> MassObject:
        host = MassObject(
            config=self.sim_config,
            mass=mass, radius=radius,
            position=pygame.Vector2(position),
            velocity=pygame.Vector2(0.0, 0.0),
            color=(255, 200, 80),
        )
        self.world.add(host)
        return host

    def _spawn_rocket_docked(self, host: MassObject, angle: float) -> Rocket:
        rocket = Rocket(
            config=self.sim_config,
            mass=self.ROCKET_MASS,
            position=pygame.Vector2(0.0, 0.0),
            velocity=pygame.Vector2(0.0, 0.0),
            alpha=self._ALPHA,
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
        rocket.dock_to(host, angle)
        return rocket

    # ------------------------------------------------------------------
    # Observação (idêntica ao DESIGN §3)
    # ------------------------------------------------------------------

    def _build_observation_for(self, rocket: Rocket,
                               host: MassObject) -> Optional[list[float]]:
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
        cos_rel = max(-1.0, min(1.0, cos_rel))
        sin_rel = (rp_x * ny - rp_y * nx) / r

        docked = 1.0 if rocket.docked_to is not None else 0.0
        fuel_frac = rocket.fuel / rocket.fuel_max if rocket.fuel_max > 0.0 else 0.0
        thrust_ratio = rocket.thrust_acceleration / a_g_ref if a_g_ref > 0.0 else 1.0
        rot_ratio = rocket.rotation_speed / omega_ref if omega_ref > 0.0 else 1.0
        log_thrust = max(-1.0, min(2.0, math.log10(max(thrust_ratio, 1e-6))))
        log_rot = max(-1.0, min(2.0, math.log10(max(rot_ratio, 1e-6))))

        return [
            r / r_ref, v_r / v_ref, v_t / v_ref,
            sin_rel, cos_rel, docked, fuel_frac,
            log_thrust, log_rot,
        ]

    # ------------------------------------------------------------------
    # Recompensa
    # ------------------------------------------------------------------

    def _compute_phi(self, i: int) -> float:
        """@brief Potencial de shaping: altitude + velocidade tangencial.

        @details Φ combina dois termos suaves para dar gradiente:
        (1) altitude, de 0 na superfície a 1 em `r_ref` (capado);
        (2) velocidade tangencial `|v_t|/v_ref`, capada em 1.
        O shaping telescopa ao longo do episódio, então não enviesa
        a política ótima — apenas torna o gradiente do GA mais
        informativo perto da solução.
        """
        rocket = self.rockets[i]
        host = self.host
        if host is None or not rocket.active or not host.active:
            return 0.0
        if host.radius <= 0.0:
            return 0.0

        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        r = math.sqrt(dx * dx + dy * dy)
        if r <= 1e-12:
            return 0.0

        # Altitude: 0 em r=R, 1 em r=r_ref, capado.
        alt_frac = (r / host.radius - 1.0) / (self._SUCCESS_R_FACTOR - 1.0)
        alt_frac = max(0.0, min(alt_frac, 1.0))

        # Tangencial: |v_t|/v_ref, capado em 1 (não premia excesso).
        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        h = dx * rvy - dy * rvx
        v_t_abs = abs(h) / r
        v_ref = self._v_ref[i]
        tan_frac = min(v_t_abs / v_ref, 1.0) if v_ref > 0.0 else 0.0

        return self._PHI_ALT_SCALE * alt_frac + self._PHI_TAN_SCALE * tan_frac

    def _update_rewards(self) -> None:
        for i in range(len(self.rockets)):
            if not self.alive[i]:
                continue
            if not self.rockets[i].active:
                self.alive[i] = False
                self.terminal_reasons[i] = "destroyed"
                continue
            self._update_reward_for(i)

    def _update_reward_for(self, i: int) -> None:
        rocket = self.rockets[i]
        host = self.host
        if host is None:
            return

        # Shaping por potencial
        new_phi = self._compute_phi(i)
        shaping = new_phi - self.prev_phi[i]
        self.prev_phi[i] = new_phi

        # Custos densos (empuxo, rotação e tempo)
        cost_thrust = self._C_THRUST if rocket.thrusting else 0.0
        cost_rotate = self._C_ROTATE * abs(rocket.rotation_input)
        self.rewards[i] += shaping - cost_thrust - cost_rotate - self._C_TIME
        self.steps[i] += 1

        if rocket.docked_to is None:
            self.ever_undocked[i] = True

        # Cinemática local para a decisão de terminal
        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        r = math.sqrt(dx * dx + dy * dy)
        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y

        terminal: Optional[float] = None
        reason: Optional[str] = None
        success_alignment: Optional[float] = None
        success_fuel_frac: Optional[float] = None
        success_time_frac: Optional[float] = None

        if r >= self._SUCCESS_R_FACTOR * host.radius:
            speed_sq = rvx * rvx + rvy * rvy
            speed = math.sqrt(speed_sq)
            if speed > 1e-6:
                h = dx * rvy - dy * rvx
                v_t = abs(h) / r
                alignment = v_t / speed
            else:
                v_t = 0.0
                alignment = 0.0

            v_ref = self._v_ref[i]
            vt_frac = (v_t / v_ref) if v_ref > 0.0 else 0.0

            # Sucesso exige alinhamento E magnitude tangencial, para
            # evitar o exploit "coast até o ápice com v_r=0 e v_t≈0".
            if (alignment >= self._SUCCESS_ALIGNMENT_MIN
                    and vt_frac >= self._SUCCESS_VT_MIN):
                fuel_frac = (
                    rocket.fuel / rocket.fuel_max
                    if rocket.fuel_max > 0.0 else 0.0
                )
                time_frac = 1.0 - (self.steps[i] / max(1, self._episode_max_steps))
                time_frac = max(0.0, time_frac)

                terminal = (
                    self._TERMINAL_SUCCESS
                    + self._ALIGNMENT_BONUS_SCALE * alignment
                    + self._FUEL_BONUS_SCALE * fuel_frac
                    + self._TIME_BONUS_SCALE * time_frac
                )
                reason = "success"
                success_alignment = alignment
                success_fuel_frac = fuel_frac
                success_time_frac = time_frac

        if terminal is None:
            if self.ever_undocked[i] and collides(host, rocket):
                speed = (host.velocity - rocket.velocity).length()
                if speed >= self.sim_config.merge_min_relative_speed:
                    terminal, reason = self._TERMINAL_CRASH, "crash"
                else:
                    terminal, reason = self._TERMINAL_REDOCKED, "fell_back"
            elif r > self._ESCAPE_R_FACTOR * host.radius:
                terminal, reason = self._TERMINAL_ESCAPE, "escape"
            elif rocket.fuel <= 0.0:
                terminal, reason = self._TERMINAL_FUEL, "fuel"
            elif self.steps[i] >= self._episode_max_steps:
                terminal, reason = self._TERMINAL_TIMEOUT, "timeout"

        if terminal is not None:
            self.rewards[i] += terminal
            self.alive[i] = False
            self.terminal_reasons[i] = reason
            if reason == "success":
                self._episode_success_alignments.append(success_alignment)
                self._episode_success_fuel_fracs.append(success_fuel_frac)
                self._episode_success_time_fracs.append(success_time_frac)

    # ------------------------------------------------------------------
    # Loop físico
    # ------------------------------------------------------------------

    def fixed_update(self, fixed_delta_time: float) -> None:
        self._episode_step += 1
        host = self.host

        if host is not None and host.active and self.networks:
            for i, (rocket, network) in enumerate(zip(self.rockets, self.networks)):
                if network is None or not rocket.active or not self.alive[i]:
                    continue
                obs = self._build_observation_for(rocket, host)
                if obs is None:
                    continue
                try:
                    rocket.control_with_network(network, obs, deterministic=True)
                except Exception as exc:
                    print(f"[ai] erro de rede: {exc}")

        bodies = self.world.entities
        self.world.fixed_update(fixed_delta_time)

        accel_model = CompositeAccelerationModel(self.gravity, bodies)
        self.integrator.step(bodies, fixed_delta_time, accel_model)

        for body in bodies:
            body.post_integrate()

        if host is not None and host.active:
            self._update_rewards()

        if (
            not any(self.alive)
            or self._episode_step >= self._episode_max_steps
            or host is None
            or not host.active
        ):
            for i in range(len(self.alive)):
                if self.alive[i]:
                    self.rewards[i] += self._TERMINAL_TIMEOUT
                    self.alive[i] = False
                    if self.terminal_reasons[i] is None:
                        self.terminal_reasons[i] = "timeout"
            self._end_episode()
            return

        self._resolve_merges()
        self.world.purge_inactive()

    def _resolve_merges(self) -> None:
        host = self.host
        if host is None or not host.active:
            return
        min_speed = self.sim_config.merge_min_relative_speed
        for i, rocket in enumerate(self.rockets):
            if not rocket.active or rocket is host or not self.alive[i]:
                continue
            if not collides(host, rocket):
                continue
            if (host.velocity - rocket.velocity).length() < min_speed:
                continue
            host.merge_with(rocket)

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _write_generation_log(self, fitness: list[float], gen_dt: float) -> None:
        host = self.host
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

        total = n * self.episodes_per_gen
        success = self._gen_terminal_counts.get("success", 0)
        success_rate = (success / total) if total > 0 else 0.0

        mean_align = (
            sum(self._gen_success_alignments) / len(self._gen_success_alignments)
            if self._gen_success_alignments else None
        )
        mean_fuel = (
            sum(self._gen_success_fuel_fracs) / len(self._gen_success_fuel_fracs)
            if self._gen_success_fuel_fracs else None
        )
        mean_time = (
            sum(self._gen_success_time_fracs) / len(self._gen_success_time_fracs)
            if self._gen_success_time_fracs else None
        )

        best_idx = max(range(n), key=lambda i: fitness[i])
        entry = {
            "event": "generation",
            "phase": "undocking",
            "gen": int(self.generation),
            "episodes_per_net": self.episodes_per_gen,
            "wall_time_s": round(gen_dt, 3),
            "host_mass": round(float(host.mass), 2),
            "host_radius": round(float(host.radius), 2),
            "population_size": n,
            "fitness": {
                "best": round(float(sorted_f[-1]), 4),
                "avg": round(float(mean_f), 4),
                "median": round(float(median_f), 4),
                "std": round(float(math.sqrt(var_f)), 4),
                "min": round(float(sorted_f[0]), 4),
            },
            "success_rate": round(float(success_rate), 4),
            "mean_alignment_at_success": (
                round(float(mean_align), 4) if mean_align is not None else None
            ),
            "mean_fuel_frac_at_success": (
                round(float(mean_fuel), 4) if mean_fuel is not None else None
            ),
            "mean_time_frac_at_success": (
                round(float(mean_time), 4) if mean_time is not None else None
            ),
            "terminals": dict(self._gen_terminal_counts),
            "best_idx": int(best_idx),
            "best_ever": (
                round(float(self._best_ever), 4)
                if self._best_ever != -math.inf else None
            ),
            "best_ever_gen": (
                int(self._best_ever_gen) if self._best_ever_gen >= 0 else None
            ),
            "individuals": [
                {"idx": i, "fitness": round(float(fitness[i]), 4)}
                for i in range(n)
            ],
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def update(self, delta_time: float) -> None:
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(
                self.input.mouse_position,
                1.1 ** self.input.mouse_wheel,
            )
        if self.input.was_pressed(pygame.K_RIGHT):
            self.speed_multiplier = min(5.0, self.speed_multiplier + 0.5)
        if self.input.was_pressed(pygame.K_LEFT):
            self.speed_multiplier = max(1.0, self.speed_multiplier - 0.5)
        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(-self.input.mouse_delta / self.camera.zoom)

    def render(self, renderer) -> None:
        alpha = self.clock.interpolation_alpha

        host = self.host
        if host is not None and host.active:
            center = self.camera.world_to_screen(host.position)
            radius = self.camera.world_to_screen_size(
                self._SUCCESS_R_FACTOR * host.radius
            )
            renderer.draw_circle((100, 200, 255), center, radius, width=1)

        self.world.render(renderer, self.camera, alpha)

        n_alive = sum(self.alive) if self.alive else 0
        n_total = len(self.alive) if self.alive else 0

        renderer.draw_text(
            f"Fase 1 — Desacoplagem   "
            f"Gen {self.generation + 1}/{self.max_generations}",
            (10, 10), size=26, color=(255, 255, 255),
        )
        renderer.draw_text(
            f"Episódio {self._gen_episodes_done + 1}/{self.episodes_per_gen}   "
            f"Vivos {n_alive}/{n_total}   step {self._episode_step}/{self._episode_max_steps}",
            (10, 42), size=22, color=(180, 220, 255),
        )
        renderer.draw_text(
            f"Velocidade = {self.speed_multiplier:.2f}x  (←/→ ajusta)",
            (10, 74), size=20, color=(140, 220, 140),
        )
        renderer.draw_text(
            f"Meta: r ≥ {self._SUCCESS_R_FACTOR:.1f}·R  "
            f"e |v_t|/|v| ≥ {self._SUCCESS_ALIGNMENT_MIN:.2f}  "
            f"e |v_t|/v_ref ≥ {self._SUCCESS_VT_MIN:.2f}",
            (10, 104), size=18, color=(255, 220, 120),
        )

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _save_best_ever(self) -> None:
        if self._best_ever_state is None:
            print("[shutdown] nenhuma geração concluída — nada a salvar.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "phase1_best.pt"
            torch.save(self._best_ever_state, path)
            print(
                f"[save] melhor rede da Fase 1: fitness {self._best_ever:+.2f} "
                f"(gen {self._best_ever_gen}) → {path}"
            )
        except Exception as exc:
            print(f"[shutdown] falha ao salvar: {exc}")

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._save_best_ever()
        super().shutdown()