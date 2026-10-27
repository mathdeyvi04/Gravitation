"""Fase 2 — Inserção orbital: elevar a órbita e mantê-la estável.

A Fase 2 reutiliza a arquitetura e o loop físico da Fase 1. A população
precisa ser inicializada com ``load_population_from_phase1`` para fazer
fine-tuning dos pesos aprendidos na desacoplagem.

Critério de sucesso: permanecer na banda de +/- 12% do raio-alvo durante
um progresso angular completo de 2*pi, como especificado em DESIGN.md.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Optional

import pygame
import torch

from src.ai.neural_population import NeuralPopulation
from src.ai.train_phase_1 import UndockingTrainer
from src.entities.rocket import Rocket
from src.physics.collision import collides
from src.physics.isolated_host_gravity import IsolatedHostGravity


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class OrbitRaiseTrainer(UndockingTrainer):
    """Fine-tuning genético para inserir e manter uma órbita circular maior.

    A classe herda de ``UndockingTrainer`` para reutilizar o integrador RK4,
    a infraestrutura Pygame, a inferência da rede e o ciclo físico. São
    substituídos o estado inicial, o potencial, os terminais, o log e o
    checkpoint para refletir o objetivo orbital da Fase 2.
    """

    _TARGET_ALPHA_RANGE = (2.0, 3.0)
    _INITIAL_RADIUS_FACTOR_RANGE = (1.50, 1.60)
    # Começa numa órbita circular inferior à meta. A orientação inicial já
    # aponta para a tangente prógrada, evitando gastar combustível em rotação
    # só para alinhar o motor antes da primeira manobra.
    _INITIAL_RADIAL_SPEED_FACTOR = 0.0
    _INITIAL_TANGENTIAL_SPEED_FACTOR = 1.0

    _ACCEPTANCE_BAND = 0.12
    _ESCAPE_R_FACTOR = 8.0
    _TIMEOUT_TARGET_ORBITS = 1.80

    # Potencial: progresso até a altitude, proximidade do raio-alvo,
    # velocidade circular e velocidade radial próxima de zero. A penalidade
    # de ultrapassagem é forte para não premiar trajetórias que escapam.
    _PHI_RADIUS_SCALE = 38.0
    _PHI_TARGET_RADIUS_SCALE = 24.0
    _PHI_TANGENTIAL_SCALE = 22.0
    _PHI_RADIAL_SCALE = 22.0
    _PHI_STABLE_ORBIT_SCALE = 22.0
    _PHI_OVERSHOOT_SCALE = 110.0
    _TARGET_RADIUS_SIGMA = 0.08
    _TANGENTIAL_SPEED_TOLERANCE = 0.20
    _RADIAL_SPEED_TOLERANCE = 0.12

    # Custos de controle calibrados para que girar/queimar continuamente seja
    # muito pior que executar manobras curtas e depois planar sem controle.
    # O combustível consumido também é penalizado explicitamente.
    _C_THRUST = 0.20
    _C_ROTATE = 0.08
    _C_THRUST_IN_BAND = 0.12
    _C_ROTATE_IN_BAND = 0.08
    _C_TIME = 0.0005
    _C_FUEL_USAGE = 250.0
    _C_RADIAL_SPEED = 0.004
    _R_STABLE_COAST_PER_STEP = 0.012

    _TERMINAL_SUCCESS = 180.0
    _TERMINAL_CRASH = -100.0
    _TERMINAL_ESCAPE = -100.0
    _TERMINAL_FUEL = -75.0
    _TERMINAL_TIMEOUT = -65.0

    _FUEL_BONUS_SCALE = 45.0
    _ORBIT_QUALITY_BONUS_SCALE = 35.0
    _TIME_BONUS_SCALE = 3.0

    def __init__(
        self,
        config,
        sim_config,
        max_generations: int = 60,
        episodes_per_gen: int = 3,
        method: str = "sexual",
        elite_count: int = 5,
        mutation_rate_start: float = 0.025,
        mutation_rate_end: float = 0.01,
        mutation_std: float = 0.03,
        base_seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> None:
        if episodes_per_gen < 1:
            raise ValueError("episodes_per_gen deve ser >= 1.")
        if max_generations < 1:
            raise ValueError("max_generations deve ser >= 1.")

        super().__init__(
            config=config,
            sim_config=sim_config,
            max_generations=max_generations,
            episodes_per_gen=episodes_per_gen,
            method=method,
            elite_count=elite_count,
            mutation_rate_start=mutation_rate_start,
            mutation_rate_end=mutation_rate_end,
            mutation_std=mutation_std,
            base_seed=base_seed,
            output_dir=output_dir,
        )
        # Arquivo separado para não misturar gerações antigas, treinadas com
        # a função de recompensa anterior, com os resultados desta versão.
        self.log_path = self.output_dir / "training_log_phase2_minfuel.jsonl"

        self._current_alpha = self._TARGET_ALPHA_RANGE[0]
        self._target_radii: list[float] = []
        self._start_radii: list[float] = []
        self._prev_angles: list[float] = []
        self._prev_fuel: list[float] = []
        self._max_radii: list[float] = []
        self._thrust_steps: list[int] = []
        self._rotation_steps: list[int] = []
        self._inside_band_steps: list[int] = []
        self._stable_progress: list[float] = []
        self._max_stable_progress: list[float] = []
        self._stable_error_sum: list[float] = []
        self._stable_radial_error_sum: list[float] = []
        self._stable_tangential_error_sum: list[float] = []
        self._stable_band_steps: list[int] = []
        self._stable_valid_steps_total: list[int] = []
        self._stable_coast_steps: list[int] = []
        self._in_band_control_steps: list[int] = []
        self._terminal_snapshots: list[Optional[dict]] = []
        self._gen_episode_records: list[dict] = []
        self._gen_alpha_values: list[float] = []

    def start_training(self, population: NeuralPopulation) -> None:
        """Inicia a Fase 2 com a população warm-start da Fase 1."""
        self._gen_episode_records = []
        self._gen_alpha_values = []
        super().start_training(population)

    def _start_episode(self) -> None:
        """Cria um host e põe os foguetes numa órbita circular inferior à meta."""
        random.seed(
            self.base_seed
            + self.generation * self.episodes_per_gen
            + self._gen_episodes_done
        )

        self.world.clear()
        self.rockets = []

        host_mass = random.uniform(*self._HOST_MASS_RANGE)
        host_radius = random.uniform(*self._HOST_RADIUS_RANGE)
        center_x = self.sim_config.world_width * 0.5
        center_y = self.sim_config.world_height * 0.5
        self.host = self._spawn_host(
            mass=host_mass,
            radius=host_radius,
            position=pygame.Vector2(center_x, center_y),
        )

        self._current_alpha = random.uniform(*self._TARGET_ALPHA_RANGE)
        self._gen_alpha_values.append(self._current_alpha)
        initial_radius = (
            random.uniform(*self._INITIAL_RADIUS_FACTOR_RANGE) * host_radius
        )
        base_angle = random.uniform(0.0, 2.0 * math.pi)
        count = len(self.networks)

        for i in range(count):
            angle = base_angle + 2.0 * math.pi * i / max(1, count)
            self._spawn_rocket_in_flight(
                host=self.host,
                angle=angle,
                radius=initial_radius,
                alpha=self._current_alpha,
            )

        self.gravity = IsolatedHostGravity(
            gravitational_constant=self.sim_config.gravitational_constant,
            softening=self.sim_config.gravitational_softening,
            host_index=0,
        )
        self.camera.set_position(self.host.position)
        self._init_episode_tracking()

    def _spawn_rocket_in_flight(
        self,
        host,
        angle: float,
        radius: float,
        alpha: float,
    ) -> Rocket:
        """Cria um foguete em órbita circular baixa, apontado para a tangente prógrada."""
        radial = pygame.Vector2(math.cos(angle), math.sin(angle))
        tangential = pygame.Vector2(-math.sin(angle), math.cos(angle))
        mu = self.sim_config.gravitational_constant * (
            host.mass + self.ROCKET_MASS
        )
        circular_speed = math.sqrt(mu / radius)
        radial_speed = self._INITIAL_RADIAL_SPEED_FACTOR * circular_speed
        tangential_speed = self._INITIAL_TANGENTIAL_SPEED_FACTOR * circular_speed
        position = host.position + radial * radius
        velocity = (
            host.velocity
            + radial * radial_speed
            + tangential * tangential_speed
        )

        rocket = Rocket(
            config=self.sim_config,
            mass=self.ROCKET_MASS,
            position=pygame.Vector2(position),
            velocity=pygame.Vector2(velocity),
            alpha=alpha,
            angle=angle + math.pi / 2.0,
            body_length=self.ROCKET_BODY_LENGTH,
            body_width=self.ROCKET_BODY_WIDTH,
            nose_length=self.ROCKET_NOSE_LENGTH,
            thrust_acceleration=self.ROCKET_THRUST_ACCELERATION,
            color=self.ROCKET_COLOR,
            nose_color=self.ROCKET_NOSE_COLOR,
        )
        self.world.add(rocket)
        self.rockets.append(rocket)
        return rocket

    def _init_episode_tracking(self) -> None:
        """Inicializa métricas e timeout em função do período da órbita-alvo."""
        count = len(self.rockets)
        self._episode_step = 0
        self.rewards = [0.0] * count
        self.alive = [True] * count
        self.steps = [0] * count
        self.ever_undocked = [True] * count
        self.terminal_reasons = [None] * count
        self.prev_phi = [0.0] * count
        self._episode_success_alignments = []
        self._episode_success_fuel_fracs = []
        self._episode_success_time_fracs = []

        self._target_radii = []
        self._start_radii = []
        self._prev_angles = []
        self._prev_fuel = []
        self._max_radii = []
        self._thrust_steps = [0] * count
        self._rotation_steps = [0] * count
        self._inside_band_steps = [0] * count
        self._stable_progress = [0.0] * count
        self._max_stable_progress = [0.0] * count
        self._stable_error_sum = [0.0] * count
        self._stable_radial_error_sum = [0.0] * count
        self._stable_tangential_error_sum = [0.0] * count
        self._stable_band_steps = [0] * count
        self._stable_valid_steps_total = [0] * count
        self._stable_coast_steps = [0] * count
        self._in_band_control_steps = [0] * count
        self._terminal_snapshots = [None] * count

        host = self.host
        if host is None or host.mass <= 0.0 or host.radius <= 0.0:
            self._v_ref = [0.0] * count
            self._episode_max_steps = 1
            return

        mu = self.sim_config.gravitational_constant * (
            host.mass + self.ROCKET_MASS
        )
        target_radius = self._current_alpha * host.radius
        target_speed = math.sqrt(mu / target_radius)
        target_period = 2.0 * math.pi * math.sqrt(target_radius**3 / mu)
        dt = max(float(self.config.fixed_timestep), 1e-9)
        self._episode_max_steps = max(
            1, int(self._TIMEOUT_TARGET_ORBITS * target_period / dt)
        )

        self._v_ref = [target_speed] * count
        for rocket in self.rockets:
            dx = rocket.position.x - host.position.x
            dy = rocket.position.y - host.position.y
            radius = math.hypot(dx, dy)
            self._target_radii.append(rocket.alpha * host.radius)
            self._start_radii.append(radius)
            self._prev_angles.append(math.atan2(dy, dx))
            self._prev_fuel.append(float(rocket.fuel))
            self._max_radii.append(radius)

        for i in range(count):
            self.prev_phi[i] = self._compute_phi(i)

    def _relative_state(self, i: int) -> tuple[float, float, float, float]:
        """Retorna (r, velocidade radial, tangencial assinada, ângulo)."""
        rocket = self.rockets[i]
        host = self.host
        if host is None:
            return 0.0, 0.0, 0.0, 0.0

        dx = rocket.position.x - host.position.x
        dy = rocket.position.y - host.position.y
        radius = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        if radius <= 1e-12:
            return 0.0, 0.0, 0.0, angle

        rvx = rocket.velocity.x - host.velocity.x
        rvy = rocket.velocity.y - host.velocity.y
        radial_velocity = (dx * rvx + dy * rvy) / radius
        tangential_velocity = (dx * rvy - dy * rvx) / radius
        return radius, radial_velocity, tangential_velocity, angle

    def _potential_for_state(
        self,
        i: int,
        radius: float,
        radial_velocity: float,
        tangential_velocity: float,
    ) -> float:
        """Potencial denso: altitude-alvo, velocidade circular e estabilidade."""
        target_radius = self._target_radii[i]
        start_radius = self._start_radii[i]
        target_speed = max(self._v_ref[i], 1e-9)
        span = max(target_radius - start_radius, 1e-9)
        progress = _clamp((radius - start_radius) / span, 0.0, 1.0)
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)

        radius_error = radius / target_radius - 1.0
        radius_quality = math.exp(
            -0.5 * min((radius_error / self._TARGET_RADIUS_SIGMA) ** 2, 60.0)
        )
        tangential_error = (
            (tangential_velocity - target_speed)
            / max(self._TANGENTIAL_SPEED_TOLERANCE * target_speed, 1e-9)
        )
        radial_error = (
            radial_velocity
            / max(self._RADIAL_SPEED_TOLERANCE * target_speed, 1e-9)
        )
        tangential_quality = math.exp(-0.5 * min(tangential_error**2, 60.0))
        radial_quality = math.exp(-0.5 * min(radial_error**2, 60.0))

        orbit_fraction = _clamp(
            self._stable_progress[i] / (2.0 * math.pi), 0.0, 1.0
        )
        overshoot = max(0.0, radius / target_radius - 1.0)
        overshoot_penalty = self._PHI_OVERSHOOT_SCALE * overshoot**2

        return (
            self._PHI_RADIUS_SCALE * smooth_progress
            + self._PHI_TARGET_RADIUS_SCALE * radius_quality
            + self._PHI_TANGENTIAL_SCALE * smooth_progress * tangential_quality
            + self._PHI_RADIAL_SCALE * smooth_progress * radial_quality
            + self._PHI_STABLE_ORBIT_SCALE * orbit_fraction
            - overshoot_penalty
        )

    def _compute_phi(self, i: int) -> float:
        radius, radial_velocity, tangential_velocity, _ = self._relative_state(i)
        if radius <= 1e-12 or not self._target_radii:
            return 0.0
        return self._potential_for_state(
            i, radius, radial_velocity, tangential_velocity
        )

    def _finish_rocket(
        self,
        i: int,
        reason: str,
        terminal_reward: float,
        orbit_quality: float = 0.0,
    ) -> None:
        """Termina um indivíduo e registra métricas no instante do terminal."""
        if not self.alive[i]:
            return

        rocket = self.rockets[i]
        target_radius = max(self._target_radii[i], 1e-9)
        radius, _, _, _ = self._relative_state(i)
        fuel_fraction = (
            _clamp(rocket.fuel / rocket.fuel_max, 0.0, 1.0)
            if rocket.fuel_max > 0.0 else 0.0
        )
        time_fraction = _clamp(
            1.0 - self.steps[i] / max(1, self._episode_max_steps), 0.0, 1.0
        )
        self.rewards[i] += terminal_reward
        self.alive[i] = False
        self.terminal_reasons[i] = reason
        self._terminal_snapshots[i] = {
            "terminal": reason,
            "fuel_remaining_frac": fuel_fraction,
            "fuel_used_frac": 1.0 - fuel_fraction,
            "radius_over_target": radius / target_radius,
            "max_radius_over_target": self._max_radii[i] / target_radius,
            "peak_orbit_progress": _clamp(
                self._max_stable_progress[i] / (2.0 * math.pi), 0.0, 1.0
            ),
            "band_step_fraction": self._inside_band_steps[i] / max(1, self.steps[i]),
            "in_band_control_fraction": (
                self._in_band_control_steps[i] / max(1, self._inside_band_steps[i])
            ),
            "stable_coast_fraction": (
                self._stable_coast_steps[i] / max(1, self._stable_valid_steps_total[i])
            ),
            "stable_valid_step_fraction": (
                self._stable_valid_steps_total[i] / max(1, self.steps[i])
            ),
            "mean_stable_radius_error": (
                self._stable_error_sum[i] / self._stable_band_steps[i]
                if self._stable_band_steps[i] else None
            ),
            "mean_stable_radial_speed_error": (
                self._stable_radial_error_sum[i] / self._stable_band_steps[i]
                if self._stable_band_steps[i] else None
            ),
            "mean_stable_tangential_speed_error": (
                self._stable_tangential_error_sum[i] / self._stable_band_steps[i]
                if self._stable_band_steps[i] else None
            ),
            "thrust_steps": self._thrust_steps[i],
            "rotation_steps": self._rotation_steps[i],
            "steps": self.steps[i],
            "orbit_quality": orbit_quality,
            "target_alpha": self._current_alpha,
        }
        if reason == "success":
            self._episode_success_alignments.append(orbit_quality)
            self._episode_success_fuel_fracs.append(fuel_fraction)
            self._episode_success_time_fracs.append(time_fraction)

    def _update_reward_for(self, i: int) -> None:
        """Aplica shaping, custos, estabilidade e condições terminais."""
        rocket = self.rockets[i]
        host = self.host
        if host is None or not rocket.active:
            return

        radius, radial_velocity, tangential_velocity, angle = self._relative_state(i)
        if radius <= 1e-12:
            self._finish_rocket(i, "crash", self._TERMINAL_CRASH)
            return

        delta_angle = math.atan2(
            math.sin(angle - self._prev_angles[i]),
            math.cos(angle - self._prev_angles[i]),
        )
        self._prev_angles[i] = angle
        self.steps[i] += 1

        target_radius = self._target_radii[i]
        target_speed = max(self._v_ref[i], 1e-9)
        in_band = abs(radius - target_radius) <= (
            self._ACCEPTANCE_BAND * target_radius
        )
        radial_speed_error = abs(radial_velocity) / target_speed
        tangential_speed_error = abs(tangential_velocity - target_speed) / target_speed
        stable_orbit_step = (
            in_band
            and tangential_velocity > 0.0
            and radial_speed_error <= self._RADIAL_SPEED_TOLERANCE
            and tangential_speed_error <= self._TANGENTIAL_SPEED_TOLERANCE
        )

        if in_band:
            self._inside_band_steps[i] += 1
            if rocket.thrusting or rocket.rotation_input != 0:
                self._in_band_control_steps[i] += 1

        if stable_orbit_step:
            # Um passo só conta para a volta de sucesso se posição e velocidade
            # já corresponderem a uma órbita aproximadamente circular.
            self._stable_progress[i] = max(
                0.0, self._stable_progress[i] + delta_angle
            )
            self._stable_error_sum[i] += abs(radius / target_radius - 1.0)
            self._stable_radial_error_sum[i] += radial_speed_error
            self._stable_tangential_error_sum[i] += tangential_speed_error
            self._stable_band_steps[i] += 1
            self._stable_valid_steps_total[i] += 1
            if not rocket.thrusting and rocket.rotation_input == 0:
                self._stable_coast_steps[i] += 1
        else:
            # Qualquer desvio de raio ou velocidade reinicia a volta estável.
            self._stable_progress[i] = 0.0
            self._stable_error_sum[i] = 0.0
            self._stable_radial_error_sum[i] = 0.0
            self._stable_tangential_error_sum[i] = 0.0
            self._stable_band_steps[i] = 0

        self._max_stable_progress[i] = max(
            self._max_stable_progress[i], self._stable_progress[i]
        )
        self._max_radii[i] = max(self._max_radii[i], radius)

        new_phi = self._potential_for_state(
            i, radius, radial_velocity, tangential_velocity
        )
        self.rewards[i] += new_phi - self.prev_phi[i]
        self.prev_phi[i] = new_phi

        if rocket.thrusting:
            self._thrust_steps[i] += 1
            self.rewards[i] -= self._C_THRUST
            if in_band:
                self.rewards[i] -= self._C_THRUST_IN_BAND
        if rocket.rotation_input:
            self._rotation_steps[i] += 1
            self.rewards[i] -= self._C_ROTATE * abs(rocket.rotation_input)
            if in_band:
                self.rewards[i] -= self._C_ROTATE_IN_BAND * abs(rocket.rotation_input)
        if stable_orbit_step and not rocket.thrusting and rocket.rotation_input == 0:
            # Recompensa por manter a órbita em voo livre: o comportamento
            # ideal é entrar na banda, cortar o motor e parar de girar.
            self.rewards[i] += self._R_STABLE_COAST_PER_STEP
        self.rewards[i] -= self._C_TIME

        fuel_consumed = max(0.0, self._prev_fuel[i] - float(rocket.fuel))
        self._prev_fuel[i] = float(rocket.fuel)
        if rocket.fuel_max > 0.0:
            self.rewards[i] -= (
                self._C_FUEL_USAGE * fuel_consumed / rocket.fuel_max
            )

        normalized_radial_speed = radial_velocity / max(self._v_ref[i], 1e-9)
        self.rewards[i] -= self._C_RADIAL_SPEED * min(
            normalized_radial_speed**2, 4.0
        )

        # Falhas físicas têm precedência sobre o sucesso orbital.
        if self._collides_with_host(rocket, host):
            self._finish_rocket(i, "crash", self._TERMINAL_CRASH)
            return
        if radius > self._ESCAPE_R_FACTOR * host.radius:
            self._finish_rocket(i, "escape", self._TERMINAL_ESCAPE)
            return
        if rocket.fuel <= 0.0:
            self._finish_rocket(i, "fuel", self._TERMINAL_FUEL)
            return
        if self.steps[i] >= self._episode_max_steps:
            self._finish_rocket(i, "timeout", self._TERMINAL_TIMEOUT)
            return

        if self._stable_progress[i] >= 2.0 * math.pi:
            steps = max(1, self._stable_band_steps[i])
            mean_radius_error = self._stable_error_sum[i] / steps
            mean_radial_error = self._stable_radial_error_sum[i] / steps
            mean_tangential_error = self._stable_tangential_error_sum[i] / steps
            radius_quality = _clamp(
                1.0 - mean_radius_error / self._ACCEPTANCE_BAND, 0.0, 1.0
            )
            radial_quality = _clamp(
                1.0 - mean_radial_error / self._RADIAL_SPEED_TOLERANCE, 0.0, 1.0
            )
            tangential_quality = _clamp(
                1.0 - mean_tangential_error / self._TANGENTIAL_SPEED_TOLERANCE,
                0.0, 1.0,
            )
            orbit_quality = _clamp(
                0.40 * radius_quality
                + 0.30 * radial_quality
                + 0.30 * tangential_quality,
                0.0, 1.0,
            )
            fuel_fraction = (
                _clamp(rocket.fuel / rocket.fuel_max, 0.0, 1.0)
                if rocket.fuel_max > 0.0 else 0.0
            )
            elapsed_fraction = _clamp(
                1.0 - self.steps[i] / max(1, self._episode_max_steps), 0.0, 1.0
            )
            terminal_reward = (
                self._TERMINAL_SUCCESS
                + self._FUEL_BONUS_SCALE * fuel_fraction
                + self._ORBIT_QUALITY_BONUS_SCALE * orbit_quality
                + self._TIME_BONUS_SCALE * elapsed_fraction
            )
            self._finish_rocket(
                i, "success", terminal_reward, orbit_quality=orbit_quality
            )

    @staticmethod
    def _collides_with_host(rocket: Rocket, host) -> bool:
        return collides(host, rocket)

    def _end_episode(self) -> None:
        """Guarda métricas por indivíduo antes que o próximo cenário seja criado."""
        host = self.host
        for i, rocket in enumerate(self.rockets):
            snapshot = self._terminal_snapshots[i]
            if snapshot is None:
                radius, _, _, _ = self._relative_state(i)
                target_radius = max(self._target_radii[i], 1e-9)
                fuel_fraction = (
                    _clamp(rocket.fuel / rocket.fuel_max, 0.0, 1.0)
                    if rocket.fuel_max > 0.0 else 0.0
                )
                snapshot = {
                    "terminal": self.terminal_reasons[i] or "timeout",
                    "fuel_remaining_frac": fuel_fraction,
                    "fuel_used_frac": 1.0 - fuel_fraction,
                    "radius_over_target": radius / target_radius,
                    "max_radius_over_target": self._max_radii[i] / target_radius,
                    "peak_orbit_progress": _clamp(
                        self._max_stable_progress[i] / (2.0 * math.pi), 0.0, 1.0
                    ),
                    "band_step_fraction": self._inside_band_steps[i] / max(1, self.steps[i]),
                    "in_band_control_fraction": (
                        self._in_band_control_steps[i] / max(1, self._inside_band_steps[i])
                    ),
                    "stable_coast_fraction": (
                        self._stable_coast_steps[i] / max(1, self._stable_valid_steps_total[i])
                    ),
                    "stable_valid_step_fraction": (
                        self._stable_valid_steps_total[i] / max(1, self.steps[i])
                    ),
                    "mean_stable_radius_error": (
                        self._stable_error_sum[i] / self._stable_band_steps[i]
                        if self._stable_band_steps[i] else None
                    ),
                    "mean_stable_radial_speed_error": (
                        self._stable_radial_error_sum[i] / self._stable_band_steps[i]
                        if self._stable_band_steps[i] else None
                    ),
                    "mean_stable_tangential_speed_error": (
                        self._stable_tangential_error_sum[i] / self._stable_band_steps[i]
                        if self._stable_band_steps[i] else None
                    ),
                    "thrust_steps": self._thrust_steps[i],
                    "rotation_steps": self._rotation_steps[i],
                    "steps": self.steps[i],
                    "orbit_quality": 0.0,
                    "target_alpha": self._current_alpha,
                }
            self._gen_episode_records.append(
                {"idx": i, "reward": float(self.rewards[i]), **snapshot}
            )

        super()._end_episode()

    def _end_generation(self) -> None:
        """Registra métricas, preserva a melhor geração e evolui a população."""
        if self.training_done:
            return

        population_size = len(self._gen_cumulative)
        divisor = max(1, self.episodes_per_gen)
        fitness = [reward / divisor for reward in self._gen_cumulative]
        elapsed = perf_counter() - self._gen_start_time

        best = max(fitness) if fitness else -math.inf
        average = sum(fitness) / len(fitness) if fitness else -math.inf
        composite = best + average if fitness else -math.inf
        if best > self._best_fitness_ever:
            self._best_fitness_ever = best
            self._best_fitness_gen = self.generation

        if (
            composite > self._best_composite_score
            and self.population is not None
            and fitness
        ):
            self._best_composite_score = composite
            self._best_composite_gen = self.generation
            best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
            self._best_composite_fitness = fitness[best_idx]
            self._best_ever_state = {
                key: tensor.detach().clone()
                for key, tensor in self.population[best_idx].state_dict().items()
            }

        self._write_generation_log(fitness, elapsed)

        records = self._gen_episode_records
        successes = [row for row in records if row["terminal"] == "success"]
        total_episodes = max(1, population_size * self.episodes_per_gen)
        success_rate = len(successes) / total_episodes
        mean_peak_orbit = _mean(
            [float(row["peak_orbit_progress"]) for row in records]
        )
        mean_max_radius = _mean(
            [float(row["max_radius_over_target"]) for row in records]
        )
        mean_fuel_used = _mean(
            [float(row["fuel_used_frac"]) for row in records]
        )
        mean_success_fuel = _mean(
            [float(row["fuel_used_frac"]) for row in successes]
        )

        print(
            f"[phase2 gen {self.generation:3d}] "
            f"best={best:+8.2f} avg={average:+8.2f} comp={composite:+8.2f}  "
            f"succ={success_rate:5.1%}  "
            f"rmax/alvo={mean_max_radius if mean_max_radius is not None else 0.0:.3f}  "
            f"órbita={mean_peak_orbit if mean_peak_orbit is not None else 0.0:.1%}  "
            f"comb={mean_fuel_used if mean_fuel_used is not None else 0.0:.1%} "
            f"(sucesso={mean_success_fuel if mean_success_fuel is not None else 0.0:.1%})  "
            f"coast={_mean([float(row['stable_coast_fraction']) for row in records]) or 0.0:.1%} "
            f"ctrl_banda={_mean([float(row['in_band_control_fraction']) for row in records]) or 0.0:.1%}  "
            f"mut={self._current_mutation_rate():.3f} dt={elapsed:.1f}s  "
            f"term={dict(self._gen_terminal_counts)}"
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
        self._gen_episode_records = []
        self._gen_alpha_values = []
        self._start_episode()

    def _write_generation_log(
        self, fitness: list[float], gen_dt: float
    ) -> None:
        """Grava uma linha JSONL com estatísticas globais e de cada indivíduo."""
        if not fitness:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(fitness)
        count = len(fitness)
        average = sum(fitness) / count
        variance = sum((value - average) ** 2 for value in fitness) / count
        median = (
            ordered[count // 2]
            if count % 2
            else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
        )
        records = self._gen_episode_records
        successes = [row for row in records if row["terminal"] == "success"]
        total = max(1, count * self.episodes_per_gen)
        best_idx = max(range(count), key=lambda i: fitness[i])

        individuals = []
        for i, score in enumerate(fitness):
            own = [row for row in records if row["idx"] == i]
            terminals = Counter(row["terminal"] for row in own)
            individuals.append({
                "idx": i,
                "fitness": round(float(score), 4),
                "successes": int(terminals.get("success", 0)),
                "terminals": dict(terminals),
                "mean_max_radius_over_target": round(
                    _mean([float(row["max_radius_over_target"]) for row in own]) or 0.0, 4
                ),
                "mean_peak_orbit_progress": round(
                    _mean([float(row["peak_orbit_progress"]) for row in own]) or 0.0, 4
                ),
                "mean_fuel_used_frac": round(
                    _mean([float(row["fuel_used_frac"]) for row in own]) or 0.0, 4
                ),
                "mean_fuel_remaining_on_success": round(
                    _mean([
                        float(row["fuel_remaining_frac"])
                        for row in own if row["terminal"] == "success"
                    ]) or 0.0, 4
                ),
                "mean_band_step_fraction": round(
                    _mean([float(row["band_step_fraction"]) for row in own]) or 0.0, 4
                ),
                "mean_in_band_control_fraction": round(
                    _mean([float(row["in_band_control_fraction"]) for row in own]) or 0.0, 4
                ),
                "mean_stable_coast_fraction": round(
                    _mean([float(row["stable_coast_fraction"]) for row in own]) or 0.0, 4
                ),
                "mean_stable_valid_step_fraction": round(
                    _mean([float(row["stable_valid_step_fraction"]) for row in own]) or 0.0, 4
                ),
                "mean_stable_radial_speed_error": round(
                    _mean([
                        float(row["mean_stable_radial_speed_error"])
                        for row in own if row["mean_stable_radial_speed_error"] is not None
                    ]) or 0.0, 4
                ),
                "mean_stable_tangential_speed_error": round(
                    _mean([
                        float(row["mean_stable_tangential_speed_error"])
                        for row in own if row["mean_stable_tangential_speed_error"] is not None
                    ]) or 0.0, 4
                ),
                "mean_thrust_steps": round(
                    _mean([float(row["thrust_steps"]) for row in own]) or 0.0, 2
                ),
                "mean_rotation_steps": round(
                    _mean([float(row["rotation_steps"]) for row in own]) or 0.0, 2
                ),
            })

        def optional_mean(rows: list[dict], key: str) -> Optional[float]:
            return _mean([float(row[key]) for row in rows])

        entry = {
            "event": "generation",
            "phase": "orbit_raise",
            "reward_profile": "minimum_fuel_coasting_v2",
            "gen": int(self.generation),
            "episodes_per_net": int(self.episodes_per_gen),
            "wall_time_s": round(float(gen_dt), 3),
            "population_size": count,
            "target_alpha_range": (
                [round(min(self._gen_alpha_values), 4), round(max(self._gen_alpha_values), 4)]
                if self._gen_alpha_values else None
            ),
            "initial_radius_factor_range": list(self._INITIAL_RADIUS_FACTOR_RANGE),
            "acceptance_band_fraction": self._ACCEPTANCE_BAND,
            "required_stable_angular_progress_rad": round(2.0 * math.pi, 6),
            "timeout_target_orbits": self._TIMEOUT_TARGET_ORBITS,
            "stable_orbit_velocity_tolerance": {
                "radial_over_target_speed": self._RADIAL_SPEED_TOLERANCE,
                "tangential_over_target_speed": self._TANGENTIAL_SPEED_TOLERANCE,
            },
            "control_costs": {
                "thrust_per_step": self._C_THRUST,
                "rotation_per_step": self._C_ROTATE,
                "extra_thrust_in_target_band": self._C_THRUST_IN_BAND,
                "extra_rotation_in_target_band": self._C_ROTATE_IN_BAND,
                "fuel_usage_scale": self._C_FUEL_USAGE,
                "stable_coasting_reward_per_step": self._R_STABLE_COAST_PER_STEP,
            },
            "fitness": {
                "best": round(float(ordered[-1]), 4),
                "avg": round(float(average), 4),
                "median": round(float(median), 4),
                "std": round(float(math.sqrt(variance)), 4),
                "min": round(float(ordered[0]), 4),
            },
            "success_rate": round(len(successes) / total, 4),
            "metrics": {
                "mean_max_radius_over_target": round(
                    optional_mean(records, "max_radius_over_target") or 0.0, 4
                ),
                "mean_peak_orbit_progress": round(
                    optional_mean(records, "peak_orbit_progress") or 0.0, 4
                ),
                "mean_fuel_used_frac": round(
                    optional_mean(records, "fuel_used_frac") or 0.0, 4
                ),
                "mean_fuel_used_frac_on_success": round(
                    optional_mean(successes, "fuel_used_frac") or 0.0, 4
                ),
                "mean_fuel_remaining_frac_on_success": round(
                    optional_mean(successes, "fuel_remaining_frac") or 0.0, 4
                ),
                "mean_band_step_fraction": round(
                    optional_mean(records, "band_step_fraction") or 0.0, 4
                ),
                "mean_in_band_control_fraction": round(
                    optional_mean(records, "in_band_control_fraction") or 0.0, 4
                ),
                "mean_stable_coast_fraction": round(
                    optional_mean(records, "stable_coast_fraction") or 0.0, 4
                ),
                "mean_stable_valid_step_fraction": round(
                    optional_mean(records, "stable_valid_step_fraction") or 0.0, 4
                ),
                "mean_stable_radial_speed_error": round(
                    optional_mean([
                        row for row in records
                        if row["mean_stable_radial_speed_error"] is not None
                    ], "mean_stable_radial_speed_error") or 0.0, 4
                ),
                "mean_stable_tangential_speed_error": round(
                    optional_mean([
                        row for row in records
                        if row["mean_stable_tangential_speed_error"] is not None
                    ], "mean_stable_tangential_speed_error") or 0.0, 4
                ),
                "mean_thrust_steps": round(
                    optional_mean(records, "thrust_steps") or 0.0, 2
                ),
                "mean_rotation_steps": round(
                    optional_mean(records, "rotation_steps") or 0.0, 2
                ),
            },
            "terminals": dict(self._gen_terminal_counts),
            "best_idx": int(best_idx),
            "best_fitness_ever": (
                round(float(self._best_fitness_ever), 4)
                if self._best_fitness_ever != -math.inf else None
            ),
            "best_fitness_gen": self._best_fitness_gen if self._best_fitness_gen >= 0 else None,
            "best_composite_score": (
                round(float(self._best_composite_score), 4)
                if self._best_composite_score != -math.inf else None
            ),
            "best_composite_gen": self._best_composite_gen if self._best_composite_gen >= 0 else None,
            "individuals": individuals,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update(self, delta_time: float) -> None:
        """Controles de câmera e velocidade da simulação (até 8x)."""
        if self.input.mouse_wheel != 0:
            self.camera.zoom_at(
                self.input.mouse_position,
                1.1 ** self.input.mouse_wheel,
            )
        if self.input.was_pressed(pygame.K_RIGHT):
            self.speed_multiplier = min(8.0, self.speed_multiplier + 0.5)
        if self.input.was_pressed(pygame.K_LEFT):
            self.speed_multiplier = max(1.0, self.speed_multiplier - 0.5)
        # Atalhos para não precisar pressionar a seta 14 vezes.
        if self.input.was_pressed(pygame.K_8):
            self.speed_multiplier = 8.0
        if self.input.was_pressed(pygame.K_1):
            self.speed_multiplier = 1.0
        if self.input.is_mouse_button_down(pygame.BUTTON_LEFT):
            self.camera.move(-self.input.mouse_delta / self.camera.zoom)

    def render(self, renderer) -> None:
        """Mostra a órbita-alvo, a banda de tolerância e o estado do treino."""
        alpha = self.clock.interpolation_alpha
        host = self.host
        if host is not None and host.active:
            center = self.camera.world_to_screen(host.position)
            target_radius = self._current_alpha * host.radius
            low_radius = target_radius * (1.0 - self._ACCEPTANCE_BAND)
            high_radius = target_radius * (1.0 + self._ACCEPTANCE_BAND)
            renderer.draw_circle(
                (100, 200, 255), center,
                self.camera.world_to_screen_size(target_radius), width=2,
            )
            renderer.draw_circle(
                (70, 130, 170), center,
                self.camera.world_to_screen_size(low_radius), width=1,
            )
            renderer.draw_circle(
                (70, 130, 170), center,
                self.camera.world_to_screen_size(high_radius), width=1,
            )

        self.world.render(renderer, self.camera, alpha)
        alive_count = sum(self.alive) if self.alive else 0
        total_count = len(self.alive) if self.alive else 0
        peak_progress = (
            max(self._max_stable_progress) / (2.0 * math.pi)
            if self._max_stable_progress else 0.0
        )

        renderer.draw_text(
            f"Fase 2 — Órbita maior   Gen {self.generation + 1}/{self.max_generations}",
            (10, 10), size=26, color=(255, 255, 255),
        )
        renderer.draw_text(
            f"Episódio {self._gen_episodes_done + 1}/{self.episodes_per_gen}   "
            f"Vivos {alive_count}/{total_count}   step {self._episode_step}/{self._episode_max_steps}",
            (10, 42), size=21, color=(180, 220, 255),
        )
        renderer.draw_text(
            f"Alvo: r = {self._current_alpha:.2f} R_host   banda ±{self._ACCEPTANCE_BAND:.0%}   "
            f"progresso estável máx. {peak_progress:.0%}",
            (10, 73), size=19, color=(255, 220, 120),
        )
        renderer.draw_text(
            f"Velocidade = {self.speed_multiplier:.2f}x (←/→; 8=8x; 1=1x)   "
            f"timeout = {self._TIMEOUT_TARGET_ORBITS:.2f} períodos-alvo",
            (10, 102), size=18, color=(140, 220, 140),
        )

    def _save_best_ever(self) -> None:
        """Salva a melhor rede encontrada na Fase 2."""
        if self._best_ever_state is None:
            print("[shutdown] Fase 2 sem geração concluída; nada a salvar.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "phase2_best.pt"
            torch.save(self._best_ever_state, path)
            print(
                f"[save] melhor rede da Fase 2: composite={self._best_composite_score:+.2f} "
                f"(gen {self._best_composite_gen}), "
                f"fitness={self._best_composite_fitness:+.2f} -> {path}"
            )
        except Exception as exc:
            print(f"[shutdown] falha ao salvar checkpoint da Fase 2: {exc}")


def load_population_from_phase1(
    checkpoint: str | Path,
    size: int,
    seed: int = 42,
    elite_count: int = 5,
    diversity_std: float = 0.02,
) -> NeuralPopulation:
    """Cria população de fine-tuning a partir de ``phase1_best.pt``.

    ``elite_count`` redes preservam exatamente os pesos da Fase 1. As demais
    recebem ruído gaussiano pequeno para iniciar uma busca evolutiva local.
    O formato de entrada esperado é o ``state_dict`` salvo pela Fase 1.
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint da Fase 1 não encontrado: {checkpoint_path}")
    if size < 1:
        raise ValueError("size deve ser >= 1.")
    if elite_count < 0 or elite_count > size:
        raise ValueError(f"elite_count deve estar entre 0 e {size}.")
    if diversity_std < 0.0:
        raise ValueError("diversity_std não pode ser negativo.")

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibilidade com versões anteriores do PyTorch
        payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"checkpoint inválido: esperado state_dict não vazio em {checkpoint_path}"
        )

    population = NeuralPopulation(size=size, seed=seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for index, network in enumerate(population):
        network.load_state_dict(state_dict, strict=True)
        network.eval()
        if index >= elite_count and diversity_std > 0.0:
            with torch.no_grad():
                for parameter in network.parameters():
                    noise = torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                    parameter.add_(noise * diversity_std)

    print(
        f"[warm-start] {elite_count} elites clonadas de {checkpoint_path}; "
        f"{size - elite_count} indivíduos diversificados (std={diversity_std:.3f})."
    )
    return population