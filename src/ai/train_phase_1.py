"""Fase 1 — Desacoplagem: aprender a sair da superfície do host.

Meta: atingir `r ≥ 1.5·R_host` com `|v_t|/|v| ≥ 0.5` e
`|v_t|/v_ref ≥ 0.5`. Bônus terminais premiam alinhamento,
combustível restante e tempo.
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
    """@brief Treina redes para desacoplar e atingir `r ≥ α·R_host`.

    @details Cada geração roda `episodes_per_gen` episódios por rede
    com host, ângulo e (se aplicável) α sorteados. A fitness é a média
    das recompensas. O GA usa torneio + crossover uniforme + mutação
    gaussiana. A rede salva é a melhor da geração de maior `best+avg`
    — o recorde absoluto pode ser um "lucky outlier" cujo filho não
    herda a qualidade.
    """

    _INITIAL_ZOOM = 0.08

    ROCKET_MASS = 40.0
    ROCKET_BODY_LENGTH = 16.0
    ROCKET_BODY_WIDTH = 7.0
    ROCKET_NOSE_LENGTH = 9.0
    ROCKET_THRUST_ACCELERATION = 1000.0
    ROCKET_COLOR = (220, 220, 230)
    ROCKET_NOSE_COLOR = (240, 110, 80)

    _HOST_MASS_RANGE = (400_000.0, 700_000.0)
    _HOST_RADIUS_RANGE = (500.0, 540.0)

    # α=1.5 encurta a subida exigida e mantém a observação em [0, 1]
    # útil (`r/r_ref = 1` exatamente na linha de sucesso).
    _ALPHA = 1.5
    _SUCCESS_R_FACTOR = _ALPHA

    _ESCAPE_R_FACTOR = 8.0

    _PHI_ALT_SCALE = 20.0
    _PHI_TAN_SCALE = 10.0

    _C_THRUST = 0.04
    _C_ROTATE = 0.005
    _C_TIME = 0.002

    _TERMINAL_SUCCESS = +100.0
    _TERMINAL_CRASH = -50.0
    _TERMINAL_REDOCKED = -50.0
    _TERMINAL_ESCAPE = -50.0
    _TERMINAL_FUEL = -30.0
    _TERMINAL_TIMEOUT = -50.0

    _SUCCESS_ALIGNMENT_MIN = 0.5
    _SUCCESS_VT_MIN = 0.5
    _ALIGNMENT_BONUS_SCALE = 50.0
    _FUEL_BONUS_SCALE = 70.0
    _TIME_BONUS_SCALE = 50.0

    _TIMEOUT_ORBITS = 0.75

    _TOURNAMENT_SIZE = 3

    _SELECTION_W_BEST = 1.0
    _SELECTION_W_AVG = 1.0

    def __init__(
        self,
        config: Configs,
        sim_config: SimulationConfig,
        max_generations: int = 50,
        episodes_per_gen: int = 3,
        method: str = "sexual",
        elite_count: int = 5,
        mutation_rate_start: float = 0.05,
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
        @param method "sexual" (crossover) ou "asexual" (clone).
        @param elite_count Indivíduos preservados intactos.
        @param mutation_rate_start Taxa de mutação na geração 0.
        @param mutation_rate_end Taxa de mutação na última geração.
        @param mutation_std Desvio-padrão do ruído gaussiano.
        @param base_seed Semente mestra.
        @param output_dir Diretório de checkpoints e log.
        """
        super().__init__(config)
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
        self.terminal_reasons: list[Optional[str]] = []
        self.prev_phi: list[float] = []
        self._v_ref: list[float] = []
        self._episode_step = 0
        self._episode_max_steps = 0
        self._episode_success_alignments: list[float] = []
        self._episode_success_fuel_fracs: list[float] = []
        self._episode_success_time_fracs: list[float] = []

        # Buffer único reusado por foguete/passo. Consumido imediatamente
        # em `_apply_network_action`, sem sobreviver à chamada.
        self._obs_buffer: list[float] = [0.0] * 9

        self._gen_cumulative: list[float] = []
        self._gen_terminal_counts: dict[str, int] = {}
        self._gen_success_alignments: list[float] = []
        self._gen_success_fuel_fracs: list[float] = []
        self._gen_success_time_fracs: list[float] = []
        self._gen_episodes_done = 0
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
        self.log_path = self.output_dir / "training_log.jsonl"

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
        """@brief Zera buffers do episódio e pré-computa `v_ref`/timeout.

        @details `prev_phi` recebe Φ do estado inicial para que o
        shaping do primeiro passo seja exato. O timeout escala com
        `_TIMEOUT_ORBITS·T_surface`.
        """
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
        """@brief Loga, evolui (ou encerra) e reconstrói o experimento.

        @details Idempotente: chamadas após `training_done=True` são
        no-op. No caminho de "max_generations atingido", não salva
        aqui — `shutdown()` cobre janela fechada também.
        """
        if self.training_done:
            return

        n = len(self._gen_cumulative)
        fitness = [r / self.episodes_per_gen for r in self._gen_cumulative]
        gen_dt = perf_counter() - self._gen_start_time

        best = max(fitness) if fitness else -math.inf
        avg = (sum(fitness) / n) if n > 0 else -math.inf
        composite = (
            self._SELECTION_W_BEST * best
            + self._SELECTION_W_AVG * avg
        ) if fitness else -math.inf

        if best > self._best_fitness_ever:
            self._best_fitness_ever = best
            self._best_fitness_gen = self.generation

        # Salva o melhor indivíduo da geração com maior `best + avg`.
        # O recorde absoluto pode ser um "lucky outlier" (bom só em
        # gen ruim) cujo filho não herda a qualidade.
        if (composite > self._best_composite_score
                and self.population is not None and fitness):
            self._best_composite_score = composite
            self._best_composite_gen = self.generation
            best_idx = max(range(n), key=lambda i: fitness[i])
            self._best_composite_fitness = fitness[best_idx]
            self._best_ever_state = {
                # `clone` — o `state_dict` original morre em `evolve`.
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
            f"best={best:+7.2f}  avg={avg:+7.2f}  comp={composite:+7.2f}  "
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
        """@brief Taxa de mutação linearmente decrescente com o progresso."""
        total = max(1, self.max_generations - 1)
        p = min(1.0, self.generation / total)
        return self.mutation_rate_start * (1.0 - p) + self.mutation_rate_end * p

    def _spawn_host(
        self,
        mass: float,
        radius: float,
        position: pygame.Vector2,
    ) -> MassObject:
        """@brief Cria e adiciona o host ao mundo."""
        host = MassObject(
            config=self.sim_config,
            mass=mass, radius=radius,
            position=pygame.Vector2(position),
            velocity=pygame.Vector2(0.0, 0.0),
            color=(255, 200, 80),
        )
        self.world.add(host)
        return host

    def _spawn_rocket_docked(
        self,
        host: MassObject,
        angle: float,
    ) -> Rocket:
        """@brief Cria um foguete já acoplado a `host` no ângulo dado."""
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
        self,
        rocket: Rocket,
        network,
        obs: list[float],
    ) -> None:
        """@brief Fast path de `control_with_network` para o hot loop.

        @details Dispensa as validações de contrato (hasattr, device,
        forma). `inference_mode` é mais rápido que `no_grad` porque o
        loop nunca treina — o GA escreve direto nos `param.data`.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            actions, _, _ = network.select_action(
                obs_tensor, deterministic=True
            )
        action = int(actions.item())
        rotation_input, thrusting = Rocket.AI_ACTIONS[action]
        rocket.set_control(rotation_input, thrusting)

    def _potential(self, r: float, v_t_abs: float, v_ref: float) -> float:
        """@brief Φ(s) — altitude normalizada + tangencial em [0, 1].

        @details Ambos os termos capados em 1 para dar gradiente suave
        sem explodir em estados extremos.
        """
        host = self.host
        if host is None or host.radius <= 0.0:
            return 0.0
        denom = self._SUCCESS_R_FACTOR - 1.0
        if denom <= 0.0:
            return 0.0
        alt_frac = (r / host.radius - 1.0) / denom
        alt_frac = max(0.0, min(alt_frac, 1.0))
        tan_frac = min(v_t_abs / v_ref, 1.0) if v_ref > 0.0 else 0.0
        return self._PHI_ALT_SCALE * alt_frac + self._PHI_TAN_SCALE * tan_frac

    def _compute_phi(self, i: int) -> float:
        """@brief Φ do foguete `i` (usado só no init de tracking)."""
        rocket = self.rockets[i]
        host = self.host
        if host is None or not rocket.active or not host.active:
            return 0.0
        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        r = math.sqrt(dx * dx + dy * dy)
        if r <= 1e-12:
            return 0.0
        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        h = dx * rvy - dy * rvx
        v_t_abs = abs(h) / r
        return self._potential(r, v_t_abs, self._v_ref[i])

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
        """@brief Recompensa e terminais do foguete `i`.

        @details Ordem: shaping → custos → desacoplagem → sucesso
        (r ≥ αR com alinhamento + vt mínimos) → colisão/fuga/fuel/
        timeout.
        """
        rocket = self.rockets[i]
        host = self.host
        if host is None:
            return

        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        r_sq = dx * dx + dy * dy
        # Guarda contra divisão por zero se o foguete "teleportar"
        # para o centro do host (fusão ou integração degenerada).
        if r_sq <= 1e-24:
            self.alive[i] = False
            self.terminal_reasons[i] = "crash"
            self.rewards[i] += self._TERMINAL_CRASH
            return
        r = math.sqrt(r_sq)

        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        h = dx * rvy - dy * rvx
        v_t_abs = abs(h) / r

        new_phi = self._potential(r, v_t_abs, self._v_ref[i])
        self.rewards[i] += new_phi - self.prev_phi[i]
        self.prev_phi[i] = new_phi

        self.rewards[i] -= self._C_THRUST if rocket.thrusting else 0.0
        self.rewards[i] -= self._C_ROTATE * abs(rocket.rotation_input)
        self.rewards[i] -= self._C_TIME
        self.steps[i] += 1

        if rocket.docked_to is None:
            self.ever_undocked[i] = True

        terminal: Optional[float] = None
        reason: Optional[str] = None
        success_alignment: Optional[float] = None
        success_fuel_frac: Optional[float] = None
        success_time_frac: Optional[float] = None

        r_host = host.radius
        if r >= self._SUCCESS_R_FACTOR * r_host:
            speed = math.sqrt(rvx * rvx + rvy * rvy)
            alignment = (v_t_abs / speed) if speed > 1e-6 else 0.0
            v_ref = self._v_ref[i]
            vt_frac = (v_t_abs / v_ref) if v_ref > 0.0 else 0.0

            # `_SUCCESS_VT_MIN` bloqueia o exploit "coast ao ápice
            # com v_r≈0 e v_t≈0" — sem ele, alignment sozinho passaria.
            if (alignment >= self._SUCCESS_ALIGNMENT_MIN
                    and vt_frac >= self._SUCCESS_VT_MIN):
                fuel_frac = (
                    rocket.fuel / rocket.fuel_max
                    if rocket.fuel_max > 0.0 else 0.0
                )
                time_frac = 1.0 - (
                    self.steps[i] / max(1, self._episode_max_steps)
                )
                if time_frac < 0.0:
                    time_frac = 0.0

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
            # `collides` é testado mesmo com o foguete acoplado: só a
            # rotação do bico pode disparar sobreposição nesse estado,
            # o que significa bico dentro do host — crash, não repouso.
            if collides(host, rocket):
                speed = (host.velocity - rocket.velocity).length()
                if not self.ever_undocked[i]:
                    terminal, reason = self._TERMINAL_CRASH, "crash_docked"
                elif speed >= self.sim_config.merge_min_relative_speed:
                    terminal, reason = self._TERMINAL_CRASH, "crash"
                else:
                    terminal, reason = self._TERMINAL_REDOCKED, "fell_back"
            elif r > self._ESCAPE_R_FACTOR * r_host:
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

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Passo físico: inferência → integração → recompensa.

        @details Encerra o episódio se todos morreram, se `host`
        ficou inativo, ou se o timeout estourou; sobreviventes nesse
        caso recebem timeout explícito.

        @param fixed_delta_time Passo fixo em segundos.
        """
        self._episode_step += 1
        host = self.host

        if host is not None and host.active and self.networks:
            for i, (rocket, network) in enumerate(zip(self.rockets, self.networks)):
                if network is None or not rocket.active or not self.alive[i]:
                    continue
                obs = self._build_observation_for(
                    rocket, host, out=self._obs_buffer
                )
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
                    self.rewards[i] += self._TERMINAL_TIMEOUT
                    self.alive[i] = False
                    if self.terminal_reasons[i] is None:
                        self.terminal_reasons[i] = "timeout"
            self._end_episode()
            return

        self._resolve_merges()
        self.world.purge_inactive()

    def _resolve_merges(self) -> None:
        """@brief Funde o host com foguetes em colisão destrutiva.

        @details Contato lento (abaixo de `merge_min_relative_speed`)
        é ignorado — é o que permite o foguete repousar no host.
        """
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
        """@brief Anexa uma linha JSONL com as métricas da geração.

        @details Formato estável — alterações quebram a análise
        histórica. Chamado antes de `evolve`, quando `self.rockets`
        ainda aponta para a geração concluída.

        @param fitness Recompensa final de cada indivíduo.
        @param gen_dt Tempo de parede da geração, em segundos.
        """
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
            "alpha": self._ALPHA,
            "success_r_factor": self._SUCCESS_R_FACTOR,
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
            "best_fitness_ever": (
                round(float(self._best_fitness_ever), 4)
                if self._best_fitness_ever != -math.inf else None
            ),
            "best_fitness_gen": (
                int(self._best_fitness_gen)
                if self._best_fitness_gen >= 0 else None
            ),
            "best_composite_score": (
                round(float(self._best_composite_score), 4)
                if self._best_composite_score != -math.inf else None
            ),
            "best_composite_gen": (
                int(self._best_composite_gen)
                if self._best_composite_gen >= 0 else None
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
        """@brief Desenha alvo, mundo e HUD do treino."""
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
            f"Velocidade = {self.speed_multiplier:.2f}x",
            (10, 74), size=20, color=(140, 220, 140),
        )
        renderer.draw_text(
            f"Meta: r ≥ {self._SUCCESS_R_FACTOR:.2f}·R  "
            f"|v_t|/|v| ≥ {self._SUCCESS_ALIGNMENT_MIN:.2f}  "
            f"|v_t|/v_ref ≥ {self._SUCCESS_VT_MIN:.2f}",
            (10, 104), size=18, color=(255, 220, 120),
        )

    def _save_best_ever(self) -> None:
        """@brief Salva em `phase1_best.pt` a melhor rede da sessão."""
        if self._best_ever_state is None:
            print("[shutdown] nenhuma geração concluída — nada a salvar.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "phase1_best.pt"
            torch.save(self._best_ever_state, path)
            print(
                f"[save] melhor rede da Fase 1:\n"
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