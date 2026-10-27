"""Fase 2 — Órbita Maior: aprender a subir suavemente e manter órbita.

Herda a habilidade de desacoplar da Fase 1 via warm-start. Cada
episódio começa acoplado e tem α sorteado no range do currículo.

Estrutura de recompensa (após correções):
- Milestones pagam progresso: undock → approach → smooth-entry →
  position → velocity → quarter → half → success.
- `smooth_entry` (+400) só dispara se o foguete entrar na r-band com
  `|v_r|/v_ref < 0.15` e **nunca tiver feito overshoot** (r > 1.25·r_ref).
  Isso força a política a aprender a subida suave, não "explode e
  desacelera".
- Overshoot é punido diretamente: -0.20/passo acima de 1.15·r_ref e
  -0.40/passo acima de 1.25·r_ref.
- Denso em banda é pequeno (0.02) para que timeout (−250) nunca
  compense o sucesso terminal (+300 + bônus).
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


class OrbitRaiseTrainer(Application):
    """@brief Fase 2 do currículo: subir a órbita suavemente e mantê-la."""

    _INITIAL_ZOOM = 0.05

    ROCKET_MASS = 40.0
    ROCKET_BODY_LENGTH = 16.0
    ROCKET_BODY_WIDTH = 7.0
    ROCKET_NOSE_LENGTH = 9.0
    ROCKET_THRUST_ACCELERATION = 1000.0
    ROCKET_COLOR = (220, 220, 230)
    ROCKET_NOSE_COLOR = (240, 110, 80)

    _HOST_MASS_RANGE = (400_000.0, 700_000.0)
    _HOST_RADIUS_RANGE = (500.0, 540.0)

    _ALPHA_MIN = 1.5
    _ALPHA_MAX_START = 1.7
    _ALPHA_MAX_END = 2.2

    _STABLE_R_BAND = 0.12
    _STABLE_V_BAND = 0.20

    _SUCCESS_ANGLE = 2.0 * math.pi

    _ESCAPE_R_FACTOR = 3.0
    _UNDOCK_R_FACTOR = 1.5

    # --- Milestones.
    # O undock vale menos que a soma total; se valesse muito, "só
    # desacoplar e morrer" seria competitivo contra "tentar chegar".
    _MILESTONE_UNDOCK = 100.0
    _MILESTONE_APPROACH = 40.0
    _MILESTONE_POSITION = 100.0
    _MILESTONE_VELOCITY = 120.0
    _MILESTONE_QUARTER = 60.0
    _MILESTONE_HALF = 120.0

    # --- Smooth entry: o sinal que ensina "subir suavemente".
    # Só dispara se o foguete chegar na r-band com |v_r| baixo e sem
    # ter feito overshoot. Uma única vez por episódio.
    _MILESTONE_SMOOTH_ENTRY = 400.0
    _SMOOTH_VR_FRAC = 0.15

    # Zona "approaching": alvo intermediário entre o undock e a banda.
    _APPROACH_R_MIN = 1.2
    _APPROACH_R_MAX_FRAC = 0.95

    # --- Overshoot. Rastreamos "passou do alvo" e punimos
    # proporcionalmente à distância em relação a r_ref. Sem isso, o
    # GA encontra "thrustar até 2× r_ref, coastar de volta" como uma
    # estratégia barata — é o comportamento que queremos eliminar.
    _OVERSHOOT_PENALTY_FACTOR = 1.15
    _OVERSHOOT_FACTOR = 1.25
    _C_OVERSHOOT = 0.20

    # --- Shaping. `Φ_VR` alto (100) torna |v_r|/v_ref caro perto de
    # r_ref — força frear cedo em vez de thrustar-e-passar.
    _PHI_R_SCALE = 60.0
    _PHI_VR_SCALE = 100.0
    _PHI_VT_SCALE = 80.0
    _PHI_CAP = 1.0

    # --- Denso. Pequeno para não ser "farmável"; o terminal precisa
    # dominar. Antes em 0.08, permitia timeout lucrativo.
    _C_IN_BAND = 0.02
    _C_IN_POSITION = 0.005

    _C_THRUST = 0.005
    _C_ROTATE = 0.001
    _C_TIME = 0.0002

    # --- Terminais. `docked_timeout` é o pior (−400) para empurrar
    # a política a tentar; timeout (−250) é pior que sucesso (+300)
    # mesmo descontando o acúmulo denso.
    _TERMINAL_SUCCESS = +300.0
    _TERMINAL_CRASH = -120.0
    _TERMINAL_REDOCKED = -30.0
    _TERMINAL_ESCAPE = -200.0
    _TERMINAL_FUEL = -80.0
    _TERMINAL_TIMEOUT = -250.0
    _TERMINAL_DOCKED_TIMEOUT = -400.0

    _FUEL_BONUS_SCALE = 60.0
    _TIME_BONUS_SCALE = 30.0

    _TIMEOUT_ORBITS = 3.0
    _MAX_EPISODE_STEPS = 10_000

    _TOURNAMENT_SIZE = 3
    _SELECTION_W_BEST = 1.0
    _SELECTION_W_AVG = 1.0

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        max_generations: int = 80,
        episodes_per_gen: int = 2,
        method: str = "sexual",
        elite_count: int = 15,
        mutation_rate_start: float = 0.04,
        mutation_rate_end: float = 0.01,
        mutation_std: float = 0.03,
        base_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        """@brief Configura o ambiente e o GA (sem popular o mundo).

        @param config Janela e loop.
        @param sim_config Física (G, softening, etc.).
        @param max_generations Total de gerações.
        @param episodes_per_gen Episódios por rede por geração.
        @param method "sexual" ou "asexual".
        @param elite_count Indivíduos preservados intactos.
        @param mutation_rate_start Taxa de mutação na geração 0.
        @param mutation_rate_end Taxa de mutação na última geração.
        @param mutation_std Desvio-padrão do ruído gaussiano.
        @param base_seed Semente mestra.
        @param output_dir Diretório de checkpoints e log.
        """
        super().__init__(config)
        self.speed_multiplier = 5.0

        self.sim_config = sim_config
        self.integrator = RK4Integrator()
        self.camera.set_zoom(self._INITIAL_ZOOM)

        self.world: World = World()
        self.host: Optional[MassObject] = None
        self.rockets: list[Rocket] = []
        self.networks: list = []
        self.population: Optional[NeuralPopulation] = None

        self.rewards: list[float] = []
        self.alive: list[bool] = []
        self.steps: list[int] = []
        self.ever_undocked: list[bool] = []
        self.ever_overshot: list[bool] = []
        self.undock_ok: list[bool] = []
        self.terminal_reasons: list[Optional[str]] = []
        self.prev_phi: list[float] = []
        self.prev_angle: list[Optional[float]] = []
        self.theta_accum: list[float] = []

        self.awarded_undock: list[bool] = []
        self.awarded_approach: list[bool] = []
        self.awarded_smooth: list[bool] = []
        self.awarded_position: list[bool] = []
        self.awarded_velocity: list[bool] = []
        self.awarded_quarter: list[bool] = []
        self.awarded_half: list[bool] = []

        self.in_band_steps: list[int] = []
        self._r_ref: list[float] = []
        self._v_ref: list[float] = []
        self._episode_step = 0
        self._episode_max_steps = 0
        self._current_alpha: float = 0.0

        self._episode_success_fuel: list[float] = []
        self._episode_success_time: list[float] = []
        self._episode_success_band: list[float] = []

        self._obs_buffer: list[float] = [0.0] * 9

        self._gen_cumulative: list[float] = []
        self._gen_terminal_counts: dict[str, int] = {}
        self._gen_undock_ok = 0
        self._gen_success_fuel: list[float] = []
        self._gen_success_time: list[float] = []
        self._gen_success_band: list[float] = []
        self._gen_episodes_done = 0
        self._gen_alphas: list[float] = []
        self._gen_start_time = 0.0

        self.generation = 0
        self.max_generations = max_generations
        self.training_done = False
        self._shutdown_done = False
        self._best_fitness_ever: float = -math.inf
        self._best_fitness_gen: int = -1
        self._best_composite_score: float = -math.inf
        self._best_composite_gen: int = -1
        self._best_composite_fitness: float = -math.inf
        self._best_ever_state: Optional[dict] = None

        self.episodes_per_gen = episodes_per_gen
        self.method = method
        self.elite_count = elite_count
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_end = mutation_rate_end
        self.mutation_std = mutation_std
        self.base_seed = base_seed

        self.output_dir = output_dir or Path("checkpoints")
        self.log_path = self.output_dir / "training_log_phase2.jsonl"

    def start_training(self, population: NeuralPopulation) -> None:
        """@brief Conecta a população inicial e inicia a geração 0."""
        self.population = population
        self.networks = list(population)
        self._gen_cumulative = [0.0] * len(population)
        self._reset_gen_state()
        self._gen_start_time = perf_counter()
        self._start_episode()

    def _reset_gen_state(self) -> None:
        self._gen_terminal_counts = {}
        self._gen_undock_ok = 0
        self._gen_success_fuel = []
        self._gen_success_time = []
        self._gen_success_band = []
        self._gen_episodes_done = 0
        self._gen_alphas = []

    def _start_episode(self) -> None:
        """@brief Limpa o mundo e monta um cenário de órbita-alvo."""
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

        self._current_alpha = random.uniform(
            self._ALPHA_MIN, self._current_alpha_hi()
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
        """@brief Zera buffers do episódio e pré-computa `v_ref`/timeout."""
        n = len(self.rockets)
        self._episode_step = 0
        self.rewards = [0.0] * n
        self.alive = [True] * n
        self.steps = [0] * n
        self.ever_undocked = [False] * n
        self.ever_overshot = [False] * n
        self.undock_ok = [False] * n
        self.terminal_reasons = [None] * n
        self.prev_phi = [0.0] * n
        self.prev_angle = [None] * n
        self.theta_accum = [0.0] * n

        self.awarded_undock = [False] * n
        self.awarded_approach = [False] * n
        self.awarded_smooth = [False] * n
        self.awarded_position = [False] * n
        self.awarded_velocity = [False] * n
        self.awarded_quarter = [False] * n
        self.awarded_half = [False] * n

        self.in_band_steps = [0] * n
        self._episode_success_fuel = []
        self._episode_success_time = []
        self._episode_success_band = []

        host = self.host
        if host is not None and host.mass > 0.0 and host.radius > 0.0:
            mu = self.sim_config.gravitational_constant * (
                host.mass + self.ROCKET_MASS
            )
            r_ref = self._current_alpha * host.radius
            v_ref = math.sqrt(mu / r_ref) if r_ref > 0.0 else 0.0
            self._r_ref = [r_ref] * n
            self._v_ref = [v_ref] * n

            T_orb = 2.0 * math.pi * math.sqrt(r_ref ** 3 / mu)
            steps_for_orbits = int(
                self._TIMEOUT_ORBITS * T_orb / self.config.fixed_timestep
            )
            self._episode_max_steps = max(
                1, min(self._MAX_EPISODE_STEPS, steps_for_orbits)
            )
        else:
            self._r_ref = [0.0] * n
            self._v_ref = [1.0] * n
            self._episode_max_steps = self._MAX_EPISODE_STEPS

        for i in range(n):
            rocket = self.rockets[i]
            if host is None or not rocket.active:
                continue
            dx = rocket.position.x - host.position.x
            dy = rocket.position.y - host.position.y
            r = math.sqrt(dx * dx + dy * dy)
            if r <= 1e-12:
                continue
            rvx = rocket.velocity.x - host.velocity.x
            rvy = rocket.velocity.y - host.velocity.y
            v_r = (dx * rvx + dy * rvy) / r
            h = dx * rvy - dy * rvx
            v_t_abs = abs(h) / r
            self.prev_phi[i] = self._potential(
                r, v_r, v_t_abs, self._r_ref[i], self._v_ref[i]
            )
            self.prev_angle[i] = math.atan2(dy, dx)

    def _end_episode(self) -> None:
        """@brief Acumula fitness do episódio; reinicia ou evolui."""
        for i in range(len(self.rockets)):
            self._gen_cumulative[i] += self.rewards[i]
            reason = self.terminal_reasons[i] or "timeout"
            self._gen_terminal_counts[reason] = (
                self._gen_terminal_counts.get(reason, 0) + 1
            )
            if self.undock_ok[i]:
                self._gen_undock_ok += 1

        self._gen_success_fuel.extend(self._episode_success_fuel)
        self._gen_success_time.extend(self._episode_success_time)
        self._gen_success_band.extend(self._episode_success_band)
        self._gen_alphas.append(self._current_alpha)
        self._gen_episodes_done += 1

        if self._gen_episodes_done >= self.episodes_per_gen:
            self._end_generation()
        else:
            self._start_episode()

    def _end_generation(self) -> None:
        """@brief Loga, evolui (ou encerra) e reconstrói o experimento."""
        if self.training_done:
            return

        n = len(self._gen_cumulative)
        fitness = [r / self.episodes_per_gen for r in self._gen_cumulative]
        gen_dt = perf_counter() - self._gen_start_time

        best = max(fitness) if fitness else -math.inf
        avg = (sum(fitness) / n) if n > 0 else -math.inf
        composite = (
            self._SELECTION_W_BEST * best + self._SELECTION_W_AVG * avg
        ) if fitness else -math.inf

        if best > self._best_fitness_ever:
            self._best_fitness_ever = best
            self._best_fitness_gen = self.generation

        if (composite > self._best_composite_score
                and self.population is not None and fitness):
            self._best_composite_score = composite
            self._best_composite_gen = self.generation
            best_idx = max(range(n), key=lambda i: fitness[i])
            self._best_composite_fitness = fitness[best_idx]
            self._best_ever_state = {
                k: v.detach().clone()
                for k, v in self.population[best_idx].state_dict().items()
            }

        self._write_generation_log(fitness, gen_dt)

        total = n * self.episodes_per_gen
        success = self._gen_terminal_counts.get("success", 0)
        success_rate = (success / total) if total > 0 else 0.0
        undock_rate = (self._gen_undock_ok / total) if total > 0 else 0.0
        avg = (sum(fitness) / n) if n > 0 else 0.0
        best = max(fitness) if fitness else 0.0

        mean_fuel = (
            sum(self._gen_success_fuel) / len(self._gen_success_fuel)
            if self._gen_success_fuel else 0.0
        )
        mean_band = (
            sum(self._gen_success_band) / len(self._gen_success_band)
            if self._gen_success_band else 0.0
        )

        alpha_hi = self._current_alpha_hi()
        print(
            f"[gen {self.generation:3d}] "
            f"α∈[{self._ALPHA_MIN:.1f},{alpha_hi:.1f}]  "
            f"best={best:+7.2f}  avg={avg:+7.2f}  comp={composite:+7.2f}  "
            f"succ={success_rate:.0%}  undock={undock_rate:.0%}  "
            f"fuel={mean_fuel:.2f}  band={mean_band:.2f}  "
            f"mut={self._current_mutation_rate():.3f}  "
            f"dt={gen_dt:.1f}s  term={self._gen_terminal_counts}"
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
        self._reset_gen_state()
        self._gen_start_time = perf_counter()
        self._start_episode()

    def _current_mutation_rate(self) -> float:
        """@brief Taxa de mutação linearmente decrescente."""
        total = max(1, self.max_generations - 1)
        p = min(1.0, self.generation / total)
        return self.mutation_rate_start * (1.0 - p) + self.mutation_rate_end * p

    def _current_alpha_hi(self) -> float:
        """@brief Topo do range de α para a geração atual."""
        progress = min(1.0, self.generation / max(1, self.max_generations - 1))
        return (
            self._ALPHA_MAX_START
            + (self._ALPHA_MAX_END - self._ALPHA_MAX_START) * progress
        )

    def _spawn_host(
        self, mass: float, radius: float, position: pygame.Vector2,
    ) -> MassObject:
        """@brief Cria e adiciona o host ao mundo."""
        host = MassObject(
            config=self.sim_config, mass=mass, radius=radius,
            position=pygame.Vector2(position),
            velocity=pygame.Vector2(0.0, 0.0),
            color=(255, 200, 80),
        )
        self.world.add(host)
        return host

    def _spawn_rocket_docked(
        self, host: MassObject, angle: float,
    ) -> Rocket:
        """@brief Cria um foguete acoplado a `host` no ângulo dado."""
        rocket = Rocket(
            config=self.sim_config,
            mass=self.ROCKET_MASS,
            position=pygame.Vector2(0.0, 0.0),
            velocity=pygame.Vector2(0.0, 0.0),
            alpha=self._current_alpha,
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

    def _build_observation_for(
        self,
        rocket: Rocket,
        host: MassObject,
        out: Optional[list[float]] = None,
    ) -> Optional[list[float]]:
        """@brief Vetor de 9 floats do DESIGN §3.

        @param rocket Foguete observado.
        @param host Corpo de referência.
        @param out Buffer opcional reutilizável (tamanho ≥ 9).
        @return Vetor preenchido, ou None em estados degenerados.
        """
        if out is None:
            out = [0.0] * 9
        if not rocket.active or not host.active:
            return None

        mu = self.sim_config.gravitational_constant * (host.mass + rocket.mass)
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
        cos_rel = max(-1.0, min(1.0, (nx * rp_x + ny * rp_y) / r))
        sin_rel = (rp_x * ny - rp_y * nx) / r

        docked = 1.0 if rocket.docked_to is not None else 0.0
        fuel_frac = rocket.fuel / rocket.fuel_max if rocket.fuel_max > 0.0 else 0.0
        thrust_ratio = rocket.thrust_acceleration / a_g_ref if a_g_ref > 0.0 else 1.0
        rot_ratio = rocket.rotation_speed / omega_ref if omega_ref > 0.0 else 1.0
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

    def _apply_network_action(
        self, rocket: Rocket, network, obs: list[float],
    ) -> None:
        """@brief Fast path de `control_with_network` para o hot loop."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            actions, _, _ = network.select_action(
                obs_tensor, deterministic=True
            )
        action = int(actions.item())
        rotation_input, thrusting = Rocket.AI_ACTIONS[action]
        rocket.set_control(rotation_input, thrusting)

    def _potential(
        self, r: float, v_r: float, v_t_abs: float,
        r_ref: float, v_ref: float,
    ) -> float:
        """@brief Φ(s): quão perto da órbita circular alvo."""
        if r_ref <= 0.0 or v_ref <= 0.0:
            return 0.0
        e_r = min(abs(r / r_ref - 1.0), self._PHI_CAP)
        e_vr = min(abs(v_r) / v_ref, self._PHI_CAP)
        e_vt = min(abs(v_t_abs / v_ref - 1.0), self._PHI_CAP)
        return -(
            self._PHI_R_SCALE * e_r
            + self._PHI_VR_SCALE * e_vr
            + self._PHI_VT_SCALE * e_vt
        )

    def _update_rewards(self) -> None:
        """@brief Itera sobre foguetes vivos e aplica recompensa/terminais."""
        for i in range(len(self.rockets)):
            if not self.alive[i]:
                continue
            if not self.rockets[i].active:
                self.alive[i] = False
                self.terminal_reasons[i] = "destroyed"
                continue
            self._update_reward_for(i)

    def _update_reward_for(self, i: int) -> None:
        """@brief Recompensa, milestones e terminais do foguete `i`.

        @details Ordem: shaping → custos → overshoot → milestones →
        terminais. A escada de milestones paga progresso incremental
        e `_MILESTONE_SMOOTH_ENTRY` só dispara se a primeira entrada
        na r-band ocorreu com `|v_r|/v_ref < _SMOOTH_VR_FRAC` e sem
        overshoot — o sinal que ensina "subida suave".
        """
        rocket = self.rockets[i]
        host = self.host
        if host is None:
            return

        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        r_sq = dx * dx + dy * dy
        if r_sq <= 1e-24:
            self.alive[i] = False
            self.terminal_reasons[i] = "crash"
            self.rewards[i] += self._TERMINAL_CRASH
            return
        r = math.sqrt(r_sq)

        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        v_r = (dx * rvx + dy * rvy) / r
        h = dx * rvy - dy * rvx
        v_t_abs = abs(h) / r

        r_ref = self._r_ref[i]
        v_ref = self._v_ref[i]

        new_phi = self._potential(r, v_r, v_t_abs, r_ref, v_ref)
        self.rewards[i] += new_phi - self.prev_phi[i]
        self.prev_phi[i] = new_phi

        self.rewards[i] -= self._C_THRUST if rocket.thrusting else 0.0
        self.rewards[i] -= self._C_ROTATE * abs(rocket.rotation_input)
        self.rewards[i] -= self._C_TIME
        self.steps[i] += 1

        if rocket.docked_to is None:
            self.ever_undocked[i] = True

        # Overshoot: rastreia "passou do alvo" e pune proporcionalmente.
        # A punição dobra acima de `_OVERSHOOT_FACTOR` — desencoraja
        # fortemente o comportamento "thrust até 2× r_ref, coastar".
        if r_ref > 0.0:
            if r > r_ref * self._OVERSHOOT_FACTOR:
                self.ever_overshot[i] = True
                self.rewards[i] -= 2.0 * self._C_OVERSHOOT
            elif r > r_ref * self._OVERSHOOT_PENALTY_FACTOR:
                self.rewards[i] -= self._C_OVERSHOOT

        # --- Milestone T1: undock.
        if not self.awarded_undock[i] and r >= self._UNDOCK_R_FACTOR * host.radius:
            self.awarded_undock[i] = True
            self.undock_ok[i] = True
            self.rewards[i] += self._MILESTONE_UNDOCK

        # --- Milestone T2: approach (passou da zona "rasa").
        if (self.ever_undocked[i]
                and not self.awarded_approach[i]
                and r >= self._APPROACH_R_MIN * host.radius
                and r < self._APPROACH_R_MAX_FRAC * r_ref):
            self.awarded_approach[i] = True
            self.rewards[i] += self._MILESTONE_APPROACH

        in_position = (
            r_ref > 0.0 and abs(r / r_ref - 1.0) < self._STABLE_R_BAND
        )
        in_velocity = (
            v_ref > 0.0
            and abs(v_r) / v_ref < self._STABLE_V_BAND
            and abs(v_t_abs / v_ref - 1.0) < self._STABLE_V_BAND
        )
        in_band = in_position and in_velocity

        vr_frac = abs(v_r) / v_ref if v_ref > 0.0 else 1.0

        # --- Milestone T2.5: smooth entry. Só dispara na PRIMEIRA
        # entrada em r-band com v_r baixo e sem overshoot prévio.
        # É o sinal mais forte contra "explode e desacelera".
        if (in_position
                and not self.awarded_smooth[i]
                and not self.ever_overshot[i]
                and vr_frac < self._SMOOTH_VR_FRAC):
            self.awarded_smooth[i] = True
            self.rewards[i] += self._MILESTONE_SMOOTH_ENTRY

        # --- Milestone T3: position.
        if in_position and not self.awarded_position[i]:
            self.awarded_position[i] = True
            self.rewards[i] += self._MILESTONE_POSITION

        # --- Milestone T4: velocity (orbit-shaped).
        if in_band and not self.awarded_velocity[i]:
            self.awarded_velocity[i] = True
            self.rewards[i] += self._MILESTONE_VELOCITY

        angle = math.atan2(dy, dx)
        prev = self.prev_angle[i]
        if prev is not None:
            delta = (angle - prev + math.pi) % (2.0 * math.pi) - math.pi
        else:
            delta = 0.0
        self.prev_angle[i] = angle

        if in_position:
            self.rewards[i] += self._C_IN_POSITION

        if in_band:
            self.rewards[i] += self._C_IN_BAND
            self.in_band_steps[i] += 1
            self.theta_accum[i] += delta

            if (not self.awarded_quarter[i]
                    and abs(self.theta_accum[i]) >= math.pi / 2):
                self.awarded_quarter[i] = True
                self.rewards[i] += self._MILESTONE_QUARTER
            if (not self.awarded_half[i]
                    and abs(self.theta_accum[i]) >= math.pi):
                self.awarded_half[i] = True
                self.rewards[i] += self._MILESTONE_HALF
        else:
            self.theta_accum[i] = 0.0

        # --- Terminais.
        terminal: Optional[float] = None
        reason: Optional[str] = None
        s_fuel = s_time = s_band = None

        if in_band and abs(self.theta_accum[i]) >= self._SUCCESS_ANGLE:
            fuel_frac = (
                rocket.fuel / rocket.fuel_max
                if rocket.fuel_max > 0.0 else 0.0
            )
            time_frac = max(
                0.0,
                1.0 - self.steps[i] / max(1, self._episode_max_steps),
            )
            band_frac = self.in_band_steps[i] / max(1, self.steps[i])
            terminal = (
                self._TERMINAL_SUCCESS
                + self._FUEL_BONUS_SCALE * fuel_frac
                + self._TIME_BONUS_SCALE * time_frac
            )
            reason = "success"
            s_fuel, s_time, s_band = fuel_frac, time_frac, band_frac

        if terminal is None:
            if collides(host, rocket):
                speed = (host.velocity - rocket.velocity).length()
                if not self.ever_undocked[i]:
                    terminal, reason = self._TERMINAL_CRASH, "crash_docked"
                elif speed >= self.sim_config.merge_min_relative_speed:
                    terminal, reason = self._TERMINAL_CRASH, "crash"
                else:
                    terminal, reason = self._TERMINAL_REDOCKED, "fell_back"
            elif r > self._ESCAPE_R_FACTOR * r_ref:
                terminal, reason = self._TERMINAL_ESCAPE, "escape"
            elif rocket.fuel <= 0.0:
                terminal, reason = self._TERMINAL_FUEL, "fuel"
            elif self.steps[i] >= self._episode_max_steps:
                if not self.ever_undocked[i]:
                    terminal = self._TERMINAL_DOCKED_TIMEOUT
                    reason = "docked_timeout"
                else:
                    terminal = self._TERMINAL_TIMEOUT
                    reason = "timeout"

        if terminal is not None:
            self.rewards[i] += terminal
            self.alive[i] = False
            self.terminal_reasons[i] = reason
            if reason == "success":
                self._episode_success_fuel.append(s_fuel)
                self._episode_success_time.append(s_time)
                self._episode_success_band.append(s_band)

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Passo físico: inferência → integração → recompensa.

        @param fixed_delta_time Passo fixo em segundos.
        """
        self._episode_step += 1
        host = self.host

        if host is not None and host.active and self.networks:
            for i, (rocket, network) in enumerate(zip(self.rockets, self.networks)):
                if network is None or not rocket.active or not self.alive[i]:
                    continue
                obs = self._build_observation_for(rocket, host, out=self._obs_buffer)
                if obs is None:
                    continue
                try:
                    self._apply_network_action(rocket, network, obs)
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
                    if not self.ever_undocked[i]:
                        self.rewards[i] += self._TERMINAL_DOCKED_TIMEOUT
                        if self.terminal_reasons[i] is None:
                            self.terminal_reasons[i] = "docked_timeout"
                    else:
                        self.rewards[i] += self._TERMINAL_TIMEOUT
                        if self.terminal_reasons[i] is None:
                            self.terminal_reasons[i] = "timeout"
                    self.alive[i] = False
            self._end_episode()
            return

        self._resolve_merges()
        self.world.purge_inactive()

    def _resolve_merges(self) -> None:
        """@brief Funde o host com foguetes em colisão destrutiva."""
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

    def _write_generation_log(self, fitness: list[float], gen_dt: float) -> None:
        """@brief Anexa uma linha JSONL com as métricas da geração."""
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
        undock_rate = (self._gen_undock_ok / total) if total > 0 else 0.0

        mean_fuel = (
            sum(self._gen_success_fuel) / len(self._gen_success_fuel)
            if self._gen_success_fuel else None
        )
        mean_time = (
            sum(self._gen_success_time) / len(self._gen_success_time)
            if self._gen_success_time else None
        )
        mean_band = (
            sum(self._gen_success_band) / len(self._gen_success_band)
            if self._gen_success_band else None
        )

        best_idx = max(range(n), key=lambda i: fitness[i])
        entry = {
            "event": "generation",
            "phase": "orbit_raise",
            "gen": int(self.generation),
            "episodes_per_net": self.episodes_per_gen,
            "wall_time_s": round(gen_dt, 3),
            "host_mass": round(float(host.mass), 2),
            "host_radius": round(float(host.radius), 2),
            "population_size": n,
            "alpha_range": [self._ALPHA_MIN, round(self._current_alpha_hi(), 4)],
            "gen_alphas": [round(a, 4) for a in self._gen_alphas],
            "fitness": {
                "best": round(float(sorted_f[-1]), 4),
                "avg": round(float(mean_f), 4),
                "median": round(float(median_f), 4),
                "std": round(float(math.sqrt(var_f)), 4),
                "min": round(float(sorted_f[0]), 4),
            },
            "success_rate": round(float(success_rate), 4),
            "undock_ok_rate": round(float(undock_rate), 4),
            "mean_fuel_frac_at_success": (
                round(float(mean_fuel), 4) if mean_fuel is not None else None
            ),
            "mean_time_frac_at_success": (
                round(float(mean_time), 4) if mean_time is not None else None
            ),
            "mean_band_frac_at_success": (
                round(float(mean_band), 4) if mean_band is not None else None
            ),
            "terminals": dict(self._gen_terminal_counts),
            "best_idx": int(best_idx),
            "best_fitness_ever": (
                round(float(self._best_fitness_ever), 4)
                if self._best_fitness_ever != -math.inf else None
            ),
            "best_fitness_gen": (
                int(self._best_fitness_gen) if self._best_fitness_gen >= 0 else None
            ),
            "best_composite_score": (
                round(float(self._best_composite_score), 4)
                if self._best_composite_score != -math.inf else None
            ),
            "best_composite_gen": (
                int(self._best_composite_gen) if self._best_composite_gen >= 0 else None
            ),
            "best_composite_fitness": (
                round(float(self._best_composite_fitness), 4)
                if self._best_composite_fitness != -math.inf else None
            ),
            "individuals": [
                {"idx": i, "fitness": round(float(fitness[i]), 4)}
                for i in range(n)
            ],
        }

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update(self, delta_time: float) -> None:
        """@brief Zoom, pan e ajuste do `speed_multiplier` (←/→)."""
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(self.input.mouse_position, 1.1 ** self.input.mouse_wheel)
        if self.input.was_pressed(pygame.K_RIGHT):
            self.speed_multiplier = min(8.0, self.speed_multiplier + 0.5)
        if self.input.was_pressed(pygame.K_LEFT):
            self.speed_multiplier = max(1.0, self.speed_multiplier - 0.5)
        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(-self.input.mouse_delta / self.camera.zoom)

    def render(self, renderer) -> None:
        """@brief Desenha alvo, banda, mundo e HUD do treino."""
        alpha = self.clock.interpolation_alpha

        host = self.host
        if host is not None and host.active and self._current_alpha > 0.0:
            center = self.camera.world_to_screen(host.position)
            r = self.camera.world_to_screen_size(self._current_alpha * host.radius)
            renderer.draw_circle((100, 200, 255), center, r, width=1)
            for f in (1.0 - self._STABLE_R_BAND, 1.0 + self._STABLE_R_BAND):
                renderer.draw_circle(
                    (80, 140, 190), center,
                    self.camera.world_to_screen_size(
                        self._current_alpha * host.radius * f
                    ),
                    width=1,
                )

        self.world.render(renderer, self.camera, alpha)

        n_alive = sum(self.alive) if self.alive else 0
        n_total = len(self.alive) if self.alive else 0

        renderer.draw_text(
            f"Fase 2 — Órbita Maior   Gen {self.generation + 1}/{self.max_generations}",
            (10, 10), size=26, color=(255, 255, 255),
        )
        renderer.draw_text(
            f"α={self._current_alpha:.2f}  "
            f"Ep {self._gen_episodes_done + 1}/{self.episodes_per_gen}  "
            f"Vivos {n_alive}/{n_total}  "
            f"step {self._episode_step}/{self._episode_max_steps}",
            (10, 42), size=22, color=(180, 220, 255),
        )
        renderer.draw_text(
            f"Velocidade = {self.speed_multiplier:.2f}x",
            (10, 74), size=20, color=(140, 220, 140),
        )
        renderer.draw_text(
            f"Meta: {self._SUCCESS_ANGLE:.2f} rad em banda  "
            f"(|Δr|/r<{self._STABLE_R_BAND}  |v|/v_ref∈1±{self._STABLE_V_BAND})",
            (10, 104), size=18, color=(255, 220, 120),
        )

    def _save_best_ever(self) -> None:
        """@brief Salva em `phase2_best.pt` a melhor rede da sessão."""
        if self._best_ever_state is None:
            print("[shutdown] nenhuma geração concluída — nada a salvar.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "phase2_best.pt"
            torch.save(self._best_ever_state, path)
            print(
                f"[save] melhor rede da Fase 2:\n"
                f"        composite = {self._best_composite_score:+.2f} "
                f"(gen {self._best_composite_gen})\n"
                f"        fitness individual = {self._best_composite_fitness:+.2f}\n"
                f"        recorde individual global = {self._best_fitness_ever:+.2f} "
                f"(gen {self._best_fitness_gen})\n"
                f"        → {path}"
            )
        except Exception as exc:
            print(f"[shutdown] falha ao salvar: {exc}")

    def shutdown(self) -> None:
        """@brief Encerra a sessão salvando a melhor rede. Idempotente."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._save_best_ever()
        super().shutdown()


def load_population_from_phase1(
    checkpoint_path: Path,
    size: int,
    seed: int,
    elite_count: int = 15,
    diversity_std: float = 0.01,
) -> NeuralPopulation:
    """@brief População warm-start a partir de um checkpoint `.pt`.

    @details Os `elite_count` primeiros carregam os pesos exatos
    (skill preservado); os demais recebem os mesmos pesos + ruído
    N(0, diversity_std²) por parâmetro.

    @param checkpoint_path Caminho do `.pt`.
    @param size Tamanho da população.
    @param seed Semente mestra (e do gerador de ruído).
    @param elite_count Indivíduos com pesos intactos.
    @param diversity_std Desvio-padrão do ruído por parâmetro.
    @return População pronta para treino.
    """
    pop = NeuralPopulation(size=size, seed=seed)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    generator = torch.Generator().manual_seed(seed + 1000)
    for i, net in enumerate(pop):
        net.load_state_dict(state)
        net.eval()
        if i < elite_count:
            continue
        with torch.no_grad():
            for param in net.parameters():
                noise = torch.randn(
                    param.shape, generator=generator, dtype=param.dtype,
                ) * diversity_std
                param.add_(noise)
    return pop