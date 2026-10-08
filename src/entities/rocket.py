import math
from typing import Optional
import pygame
from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.massobject import MassObject


class Rocket(MassObject):
    """Corpo controlável com orientação, rotação e propulsão."""

    __slots__ = (
        "angle",
        "previous_angle",
        "rotation_input",
        "rotation_speed",
        "thrust_acceleration",
        "thrusting",
        "body_length",
        "body_width",
        "nose_length",
        "nose_color",
    )

    def __init__(
        self,
        config: SimulationConfig,
        mass: float,
        position: pygame.Vector2,
        velocity: Optional[pygame.Vector2] = None,
        angle: float = -math.pi * 0.5,
        body_length: float = 34.0,
        body_width: float = 16.0,
        nose_length: float = 18.0,
        rotation_speed: float = math.radians(120.0),
        thrust_acceleration: float = 90.0,
        color: tuple[int, int, int] = (220, 220, 230),
        nose_color: tuple[int, int, int] = (240, 110, 80),
    ) -> None:
        total_length = body_length * 0.5 + nose_length
        bounding_radius = math.hypot(
            total_length,
            body_width * 0.5,
        )

        if velocity is None:
            velocity = pygame.Vector2(0.0, 0.0)

        super().__init__(
            config=config,
            mass=mass,
            radius=bounding_radius,
            position=position,
            velocity=velocity,
            color=color,
        )

        self.angle = float(angle)
        self.previous_angle = self.angle

        self.rotation_input = 0
        self.rotation_speed = float(rotation_speed)

        self.thrust_acceleration = float(thrust_acceleration)
        self.thrusting = False

        self.body_length = float(body_length)
        self.body_width = float(body_width)
        self.nose_length = float(nose_length)

        self.nose_color = nose_color

    @property
    def forward(self) -> pygame.Vector2:
        """Direção do bico em world_space."""
        return pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle),
        )

    def set_control(
        self,
        rotation_input: int,
        thrusting: bool,
    ) -> None:
        self.rotation_input = max(
            -1,
            min(1, int(rotation_input)),
        )
        self.thrusting = bool(thrusting)

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Atualiza a orientação usando o timestep fixo."""

        self.previous_angle = self.angle

        self.angle += (
            self.rotation_input
            * self.rotation_speed
            * fixed_delta_time
        )

        self.angle = math.atan2(
            math.sin(self.angle),
            math.cos(self.angle),
        )

    def additional_acceleration(
        self,
        position: pygame.Vector2,
        velocity: pygame.Vector2,
    ) -> Optional[pygame.Vector2]:
        """Retorna a aceleração produzida pelo motor."""

        if (
            not self.thrusting
            or self.thrust_acceleration <= 0.0
        ):
            return None

        return self.forward * self.thrust_acceleration

    def prediction_acceleration(
        self,
        elapsed_time: float,
    ) -> Optional[pygame.Vector2]:
        """Aceleração do motor durante uma previsão futura."""

        if (
            not self.thrusting
            or self.thrust_acceleration <= 0.0
        ):
            return None

        predicted_angle = (
            self.angle
            + self.rotation_input
            * self.rotation_speed
            * elapsed_time
        )

        return pygame.Vector2(
            math.cos(predicted_angle),
            math.sin(predicted_angle),
        ) * self.thrust_acceleration

    def _interpolated_angle(self, alpha: float) -> float:
        if alpha >= 1.0:
            return self.angle

        delta = (
            self.angle
            - self.previous_angle
            + math.pi
        ) % (2.0 * math.pi) - math.pi

        return self.previous_angle + delta * alpha

    def _world_body_rectangle(
        self,
        position: pygame.Vector2,
        angle: float,
    ) -> list[pygame.Vector2]:
        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5

        local_points = (
            pygame.Vector2(-half_length, -half_width),
            pygame.Vector2(half_length, -half_width),
            pygame.Vector2(half_length, half_width),
            pygame.Vector2(-half_length, half_width),
        )

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return [
            pygame.Vector2(
                position.x
                + point.x * cos_angle
                - point.y * sin_angle,
                position.y
                + point.x * sin_angle
                + point.y * cos_angle,
            )
            for point in local_points
        ]

    def _world_nose_triangle(
        self,
        position: pygame.Vector2,
        angle: float,
    ) -> list[pygame.Vector2]:
        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5

        local_points = (
            pygame.Vector2(
                half_length,
                -half_width,
            ),
            pygame.Vector2(
                half_length + self.nose_length,
                0.0,
            ),
            pygame.Vector2(
                half_length,
                half_width,
            ),
        )

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return [
            pygame.Vector2(
                position.x
                + point.x * cos_angle
                - point.y * sin_angle,
                position.y
                + point.x * sin_angle
                + point.y * cos_angle,
            )
            for point in local_points
        ]

    def _world_polygon(
        self,
        position: pygame.Vector2,
        angle: float,
    ) -> list[pygame.Vector2]:
        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5
        tip_x = half_length + self.nose_length

        local_points = (
            pygame.Vector2(-half_length, -half_width),
            pygame.Vector2(half_length, -half_width),
            pygame.Vector2(tip_x, 0.0),
            pygame.Vector2(half_length, half_width),
            pygame.Vector2(-half_length, half_width),
        )

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return [
            pygame.Vector2(
                position.x
                + point.x * cos_angle
                - point.y * sin_angle,
                position.y
                + point.x * sin_angle
                + point.y * cos_angle,
            )
            for point in local_points
        ]

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        position = (
            self.previous_position.lerp(
                self.position,
                alpha,
            )
            if alpha < 1.0
            else self.position
        )

        angle = self._interpolated_angle(alpha)

        if not self.is_visible(camera, alpha):
            return

        body = [
            camera.world_to_screen(point)
            for point in self._world_body_rectangle(
                position,
                angle,
            )
        ]

        nose = [
            camera.world_to_screen(point)
            for point in self._world_nose_triangle(
                position,
                angle,
            )
        ]

        renderer.draw_polygon(
            self.color,
            body,
        )

        renderer.draw_polygon(
            self.nose_color,
            nose,
        )

    def get_world_bounds(
        self,
        alpha: float = 1.0,
    ) -> tuple[float, float, float, float]:
        position = (
            self.previous_position.lerp(
                self.position,
                alpha,
            )
            if alpha < 1.0
            else self.position
        )

        angle = self._interpolated_angle(alpha)

        points = self._world_polygon(
            position,
            angle,
        )

        xs = [point.x for point in points]
        ys = [point.y for point in points]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    def _to_local(
        self,
        world_position: pygame.Vector2,
        alpha: float,
    ) -> pygame.Vector2:
        position = (
            self.previous_position.lerp(
                self.position,
                alpha,
            )
            if alpha < 1.0
            else self.position
        )

        angle = self._interpolated_angle(alpha)

        delta = world_position - position

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return pygame.Vector2(
            delta.x * cos_angle
            + delta.y * sin_angle,
            -delta.x * sin_angle
            + delta.y * cos_angle,
        )

    @staticmethod
    def _point_in_triangle(
        point: pygame.Vector2,
        a: pygame.Vector2,
        b: pygame.Vector2,
        c: pygame.Vector2,
    ) -> bool:
        def cross(
            p1: pygame.Vector2,
            p2: pygame.Vector2,
        ) -> float:
            return (
                p1.x * p2.y
                - p1.y * p2.x
            )

        d1 = cross(
            b - a,
            point - a,
        )
        d2 = cross(
            c - b,
            point - b,
        )
        d3 = cross(
            a - c,
            point - c,
        )

        has_negative = (
            d1 < 0.0
            or d2 < 0.0
            or d3 < 0.0
        )

        has_positive = (
            d1 > 0.0
            or d2 > 0.0
            or d3 > 0.0
        )

        return not (
            has_negative
            and has_positive
        )

    @staticmethod
    def _distance_to_segment(
        point: pygame.Vector2,
        a: pygame.Vector2,
        b: pygame.Vector2,
    ) -> float:
        segment = b - a
        length_squared = segment.length_squared()

        if length_squared <= 1e-12:
            return point.distance_to(a)

        t = max(
            0.0,
            min(
                1.0,
                (point - a).dot(segment)
                / length_squared,
            ),
        )

        projection = a + segment * t

        return point.distance_to(projection)

    def hit_test(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> bool:
        point = self._to_local(
            world_position,
            alpha,
        )

        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5

        if (
            -half_length <= point.x <= half_length
            and -half_width <= point.y <= half_width
        ):
            return True

        a = pygame.Vector2(
            half_length,
            -half_width,
        )

        b = pygame.Vector2(
            half_length + self.nose_length,
            0.0,
        )

        c = pygame.Vector2(
            half_length,
            half_width,
        )

        return self._point_in_triangle(
            point,
            a,
            b,
            c,
        )

    def click_distance(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> float:
        if self.hit_test(
            world_position,
            alpha,
        ):
            return 0.0

        point = self._to_local(
            world_position,
            alpha,
        )

        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5
        tip_x = half_length + self.nose_length

        polygon = (
            pygame.Vector2(
                -half_length,
                -half_width,
            ),
            pygame.Vector2(
                half_length,
                -half_width,
            ),
            pygame.Vector2(
                tip_x,
                0.0,
            ),
            pygame.Vector2(
                half_length,
                half_width,
            ),
            pygame.Vector2(
                -half_length,
                half_width,
            ),
        )

        return min(
            self._distance_to_segment(
                point,
                polygon[i],
                polygon[
                    (i + 1) % len(polygon)
                ],
            )
            for i in range(len(polygon))
        )