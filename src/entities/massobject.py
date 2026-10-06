import math
import pygame
import random
from typing import Optional
from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity

class MassObject(Entity):

    """Corpo com massa e raio que interage gravitacionalmente.

    Forças são acumuladas em `force` por um sistema externo (por exemplo,
    chamando `apply_mutual_gravity` em cada par de corpos) e integradas
    em `fixed_update`. O acumulador é zerado após cada integração.

    Interpolação: `previous_position` guarda o estado imediatamente
    anterior ao último passo fixo, permitindo render suave entre passos.
    """

    __slots__ = (
        "mass",
        "radius",
        "color",
        "force",
        "last_force",
        "previous_position",
    )

    def __init__(
            self,
            config: SimulationConfig,
            mass: Optional[float] = None,
            radius: Optional[float] = None,
            position: Optional[pygame.Vector2] = None,
            velocity: Optional[pygame.Vector2] = None,
            color: tuple[int, int, int] = (220, 220, 220),
    ) -> None:
        """Cria um corpo. Qualquer parâmetro deixado como None é sorteado.

        `config` fornece os limites do sorteio (mundo, massa, raio,
        velocidade). Sem ele, caem em valores padrão razoáveis e a
        geração aleatória fica limitada a esses defaults — útil apenas
        para testes rápidos.
        """
        if config is None:
            world_w, world_h = 1000.0, 1000.0
            mass_range = (1.0, 10.0)
            radius_range = (4.0, 16.0)
            speed_range = (0.0, 50.0)
        else:
            world_w = config.world_width
            world_h = config.world_height
            mass_range = config.mass_range
            radius_range = config.radius_range
            speed_range = config.velocity_range

        if position is None:
            position = pygame.Vector2(
                random.uniform(0.0, world_w),
                random.uniform(0.0, world_h),
            )

        if mass is None:
            mass = random.uniform(*mass_range)

        if radius is None:
            # Deriva o raio da massa se o usuário só informou a massa.
            # r ∝ m^(1/3) preserva densidade uniforme entre corpos.
            radius = radius_range[0] + (
                    (radius_range[1] - radius_range[0])
                    * ((mass - mass_range[0]) / (mass_range[1] - mass_range[0]))
            ) ** (1.0 / 3.0) if mass_range[1] > mass_range[0] else radius_range[0]

        if velocity is None:
            speed = random.uniform(*speed_range)
            angle = random.uniform(0.0, 2.0 * math.pi)
            velocity = pygame.Vector2(
                speed * math.cos(angle),
                speed * math.sin(angle),
            )

        # Com isso aqui criamos outros atributos
        super().__init__(position)
        if velocity is not None:
            self.velocity = pygame.Vector2(velocity)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.force = pygame.Vector2(0.0, 0.0)
        self.previous_position = pygame.Vector2(self.position)

        # Vamos guardar informações para podermos desenhá-las
        self.last_force = pygame.Vector2(0.0, 0.0)

    @property
    def inv_mass(self) -> float:
        """Inverso da massa (0.0 se massless). Útil em integração."""
        return 1.0 / self.mass if self.mass > 0.0 else 0.0

    def apply_mutual_gravity(
        self,
        other: "MassObject",
        gravitational_constant: float,
        softening: float = 0.0,
    ) -> None:
        """Aplica a atração gravitacional mútua entre `self` e `other`.

        Respeita a 3ª lei de Newton: mesma magnitude aplicada em cada
        corpo, sentidos opostos. `softening` (ε) evita singularidade
        quando a distância tende a zero (modelo de Plummer):
            F = G·m₁·m₂ / (r² + ε²)^(3/2) · Δ
        """
        delta = other.position - self.position
        dist_sq = delta.length_squared() + softening * softening
        if dist_sq < 1e-12:
            return

        inv_dist = 1.0 / math.sqrt(dist_sq)
        scale = gravitational_constant * self.mass * other.mass * inv_dist / dist_sq
        force = delta * scale
        self.force += force
        other.force -= force

    def update(self, delta_time: float) -> None:
        """MassObject não usa delta variável: toda física é em passo fixo."""

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Integra o movimento (Euler semi-implícito) e zera a força.

        A ordem — atualizar velocidade, depois posição com a velocidade
        nova — corresponde ao integrador semi-implícito, que é estável
        para sistemas gravitacionais simples.
        """
        self.previous_position.update(self.position)
        if self.mass > 0.0:
            self.velocity += self.force * (fixed_delta_time / self.mass)
        self.position += self.velocity * fixed_delta_time
        self.last_force.update(self.force)
        self.force.update(0.0, 0.0)

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        """Desenha o corpo como um círculo preenchido."""
        pos = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )
        renderer.draw_circle(
            self.color,
            camera.world_to_screen(pos),
            camera.escalar_world_to_screen(self.radius),
        )

    def overlaps(self, other: "MassObject") -> bool:
        """Indica se os discos de `self` e `other` se sobrepõem. Útil para vermos a colisão."""
        r = self.radius + other.radius
        return self.position.distance_squared_to(other.position) <= r * r

    def merge_with(self, other: "MassObject") -> None:
        """Absorve `other` conservando massa, momento e volume.

        O centro de massa e o momento linear total são preservados; o
        raio é recalculado assumindo densidade uniforme (r ∝ m^(1/3)).
        `other` é destruído ao final. Idempotente se `other` já estiver
        inativo, porque `destroy()` é idempotente.
        """
        total = self.mass + other.mass
        if total <= 0.0:
            other.destroy()
            return

        self.position = (
            self.position * self.mass + other.position * other.mass
        ) / total
        self.velocity = (
            self.velocity * self.mass + other.velocity * other.mass
        ) / total
        self.radius = (self.radius ** 3 + other.radius ** 3) ** (1.0 / 3.0)
        self.mass = total
        other.destroy()