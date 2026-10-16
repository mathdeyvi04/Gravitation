import math
from typing import Optional, Sequence

import pygame

from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.massobject import MassObject


class Rocket(MassObject):
    """Corpo controlável com orientação, rotação e propulsão.

    O foguete pode estar **acoplado** a um hospedeiro (`docked_to`).
    Enquanto acoplado, sua posição e velocidade acompanham o hospedeiro
    a cada passo fixo e a rotação é fixada na direção radial — como se
    estivesse pousado na superfície. Qualquer comando do jogador
    (rotação ou empuxo) desacopla.
    """

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
        "docked_to",
        "docking_angle",
    )

    ROCKET_COLOR = (220, 220, 230)
    NOSE_COLOR = (240, 110, 80)

    # Folga entre a superfície do hospedeiro e a extremidade traseira
    # do foguete, em unidades de mundo. Evita que os círculos de
    # colisão se toquem tangencialmente logo no primeiro frame.
    DOCKING_MARGIN = 2.0

    def __init__(
        self,
        config: SimulationConfig,
        mass: float,
        position: pygame.Vector2,
        velocity: Optional[pygame.Vector2] = None,
        angle: float = -math.pi * 0.5,
        body_length: float = 24.0,
        body_width: float = 10.0,
        nose_length: float = 12.0,
        rotation_speed: float = math.radians(120.0),
        thrust_acceleration: float = 120.0,
        color: tuple[int, int, int] = ROCKET_COLOR,
        nose_color: tuple[int, int, int] = NOSE_COLOR,
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

        self.docked_to: Optional[MassObject] = None
        self.docking_angle: float = 0.0

    # -- Acoplamento ---------------------------------------------------

    def dock_to(self, host: MassObject, angle: float) -> None:
        """Acopla o foguete à superfície de `host`, no ângulo `angle`.

        `angle` é a direção radial (em world_space) do centro do
        hospedeiro até o foguete. O bico nasce apontando nessa mesma
        direção — o foguete aparece "de pé" sobre a superfície.

        A transição é tratada como uma colisão perfeitamente
        inelástica: o par passa a compartilhar a velocidade do centro
        de massa, preservando o momento linear do sistema. Sem isso,
        o foguete simplesmente adotaria a velocidade do hospedeiro e
        o resíduo `m_r · (v_r − v_h)` seria descartado — invisível
        para planetas, mas mensurável para luas e meteoros.
        """
        self.docked_to = host
        self.docking_angle = float(angle)
        self.angle = self.docking_angle
        self.previous_angle = self.angle

        total_mass = self.mass + host.mass
        if total_mass > 0.0:
            common_velocity = (
                                      self.velocity * self.mass + host.velocity * host.mass
                              ) / total_mass
            self.velocity.update(common_velocity)
            host.velocity.update(common_velocity)

        self._sync_to_host()

    def undock(self) -> None:
        """Desacopla o foguete, deixando-o sob controle pleno."""
        self.docked_to = None

    def _sync_to_host(self) -> None:
        """Ajusta posição e velocidade para o estado do hospedeiro.

        Não toca em `self.angle` — a orientação é controlada pelo
        jogador via `fixed_update`, mesmo enquanto acoplado. É o que
        permite girar o foguete sobre a superfície sem que ele caia.
        """
        host = self.docked_to
        if host is None:
            return

        radial = pygame.Vector2(
            math.cos(self.docking_angle),
            math.sin(self.docking_angle),
        )
        distance = host.radius + self.body_length * 0.5 + self.DOCKING_MARGIN

        self.position.update(host.position + radial * distance)
        self.previous_position.update(self.position)
        self.velocity.update(host.velocity)

    def post_integrate(self) -> None:
        """Re-sincroniza com o hospedeiro após o passo de integração.

        RK4 escreve em `position` e `velocity` uniformemente, sem saber
        do acoplamento. Reposicionamos aqui para que o foguete continue
        “grudado” na superfície. Se o hospedeiro foi destruído (por
        fusão), desacopla e segue livre.
        """
        if self.docked_to is None:
            return
        if not self.docked_to.active:
            self.docked_to = None
            return
        self._sync_to_host()

    # -- Física --------------------------------------------------------

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
        """Aplica comandos do jogador.

        Girar enquanto acoplado apenas muda a orientação do foguete —
        ele permanece pousado. Só o empuxo desacopla. Isso permite ao
        jogador mirar o bico antes de decolar, sem risco de cair sobre
        o planeta por causa do tempo gasto girando.
        """
        thrusting = bool(thrusting)
        if self.docked_to is not None and thrusting and not self.thrusting:
            self._sync_to_host()
            self.undock()

        self.rotation_input = max(-1, min(1, int(rotation_input)))
        self.thrusting = thrusting

    def try_land(
        self,
        bodies: Sequence[MassObject],
        max_relative_speed: float,
    ) -> bool:
        """Acopla o foguete a um corpo quando ele cruza a altitude de acoplagem.

        O gatilho é a casca ``r_dock = host.radius + body_length/2 +
        DOCKING_MARGIN`` — a mesma altitude em que `attach_coefficient`
        prevê o impacto e em que `_sync_to_host` repousa o foguete.
        ``collides`` seria enganoso aqui: a forma de colisão do foguete
        toca o hospedeiro a ``R + body_length/2 − body_width/2``, duas
        unidades *abaixo* de ``r_dock``. Nessas duas unidades, a
        gravidade deposita dezenas de u/s, então um foguete que chega a
        ``r_dock`` dentro do envelope já cruzou o limiar de fusão quando
        ``collides`` dispara. Checar em ``r_dock`` alinha o gatilho ao
        coeficiente exibido no inspector.

        Custo por corpo, no laço quente:

        - **1 ``Vector2``** — ``delta``, o deslocamento até o corpo,
          reaproveitado no ``atan2`` final. Todo o resto é aritmética
          de ``float``.
        - **0 alocações** para corpos que falham no teste de distância
          (o caso comum).
        - **0 alocações** para o teste de velocidade, que compara
          ``dvx² + dvy²`` em componentes — sem ``Vector2`` temporário.

        Devolve ``True`` se acoplou. Corpos inativos, o próprio foguete
        e coincidências de posição são ignorados; se algum candidato
        está dentro da casca mas acima de `max_relative_speed`, a
        checagem continua nos demais — o foguete pode passar raspando
        por um corpo a alta velocidade e ainda pousar em outro.
        """
        if self.docked_to is not None:
            return False

        # Termo constante do raio de acoplagem. `r_dock = body.radius +
        # self._dock_offset`, e só `body.radius` muda entre corpos.
        dock_offset = self.body_length * 0.5 + self.DOCKING_MARGIN
        max_speed_sq = max_relative_speed * max_relative_speed

        for body in bodies:
            if body is self or not body.active:
                continue

            delta = self.position - body.position
            distance_squared = delta.length_squared()

            r_dock = body.radius + dock_offset
            if distance_squared > r_dock * r_dock:
                continue

            if distance_squared < 1e-12:
                continue

            dvx = self.velocity.x - body.velocity.x
            dvy = self.velocity.y - body.velocity.y
            if dvx * dvx + dvy * dvy > max_speed_sq:
                continue

            self.dock_to(body, math.atan2(delta.y, delta.x))
            return True

        return False

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Atualiza a orientação própria do foguete.

        Roda tanto acoplado quanto livre: o jogador precisa poder mirar
        o bico antes de decolar. A posição, quando acoplado, é
        restaurada em `post_integrate` por `_sync_to_host`.
        """
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
        if self.docked_to is not None:
            return None
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
        if self.docked_to is not None:
            return None
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

    # -- Colisão multi-círculo -----------------------------------------

    def collision_circles(
        self,
        alpha: float = 1.0,
    ) -> tuple[tuple[pygame.Vector2, float], ...]:
        """Quatro círculos alinhados ao eixo do foguete.

        Todos têm raio igual à meia-largura do corpo, então tangenciam
        as laterais do retângulo. Quatro centros cobrem a extensão
        total (traseira, meio, frente, bico) sem deixar vão.
        """
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )
        angle = self._interpolated_angle(alpha)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        half_length = self.body_length * 0.5
        radius = self.body_width * 0.5

        local_xs = (
            -half_length + radius,
            0.0,
            half_length - radius,
            half_length + self.nose_length - radius,
        )

        return tuple(
            (
                pygame.Vector2(
                    position.x + lx * cos_a,
                    position.y + lx * sin_a,
                ),
                radius,
            )
            for lx in local_xs
        )

    # -- Interpolação e geometria auxiliar ----------------------------

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
                position.x + point.x * cos_angle - point.y * sin_angle,
                position.y + point.x * sin_angle + point.y * cos_angle,
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
            pygame.Vector2(half_length, -half_width),
            pygame.Vector2(half_length + self.nose_length, 0.0),
            pygame.Vector2(half_length, half_width),
        )

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return [
            pygame.Vector2(
                position.x + point.x * cos_angle - point.y * sin_angle,
                position.y + point.x * sin_angle + point.y * cos_angle,
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
                position.x + point.x * cos_angle - point.y * sin_angle,
                position.y + point.x * sin_angle + point.y * cos_angle,
            )
            for point in local_points
        ]

    # -- Render --------------------------------------------------------

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        angle = self._interpolated_angle(alpha)

        if not self.is_visible(camera, alpha):
            return

        body = [
            camera.world_to_screen(point)
            for point in self._world_body_rectangle(position, angle)
        ]

        nose = [
            camera.world_to_screen(point)
            for point in self._world_nose_triangle(position, angle)
        ]

        renderer.draw_polygon(self.color, body)
        renderer.draw_polygon(self.nose_color, nose)

    def get_world_bounds(
        self,
        alpha: float = 1.0,
    ) -> tuple[float, float, float, float]:
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )

        angle = self._interpolated_angle(alpha)
        points = self._world_polygon(position, angle)

        xs = [point.x for point in points]
        ys = [point.y for point in points]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    # -- Picking -------------------------------------------------------

    def _to_local(
        self,
        world_position: pygame.Vector2,
        alpha: float,
    ) -> pygame.Vector2:
        position = (
            self.previous_position.lerp(self.position, alpha)
            if alpha < 1.0
            else self.position
        )
        angle = self._interpolated_angle(alpha)
        delta = world_position - position

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        return pygame.Vector2(
            delta.x * cos_angle + delta.y * sin_angle,
            -delta.x * sin_angle + delta.y * cos_angle,
        )

    @staticmethod
    def _point_in_triangle(
        point: pygame.Vector2,
        a: pygame.Vector2,
        b: pygame.Vector2,
        c: pygame.Vector2,
    ) -> bool:
        def cross(p1: pygame.Vector2, p2: pygame.Vector2) -> float:
            return p1.x * p2.y - p1.y * p2.x

        d1 = cross(b - a, point - a)
        d2 = cross(c - b, point - b)
        d3 = cross(a - c, point - c)

        has_negative = d1 < 0.0 or d2 < 0.0 or d3 < 0.0
        has_positive = d1 > 0.0 or d2 > 0.0 or d3 > 0.0

        return not (has_negative and has_positive)

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
                (point - a).dot(segment) / length_squared,
            ),
        )

        projection = a + segment * t
        return point.distance_to(projection)

    def hit_test(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> bool:
        point = self._to_local(world_position, alpha)

        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5

        if (
            -half_length <= point.x <= half_length
            and -half_width <= point.y <= half_width
        ):
            return True

        a = pygame.Vector2(half_length, -half_width)
        b = pygame.Vector2(half_length + self.nose_length, 0.0)
        c = pygame.Vector2(half_length, half_width)

        return self._point_in_triangle(point, a, b, c)

    def click_distance(
        self,
        world_position: pygame.Vector2,
        alpha: float = 1.0,
    ) -> float:
        if self.hit_test(world_position, alpha):
            return 0.0

        point = self._to_local(world_position, alpha)

        half_length = self.body_length * 0.5
        half_width = self.body_width * 0.5
        tip_x = half_length + self.nose_length

        polygon = (
            pygame.Vector2(-half_length, -half_width),
            pygame.Vector2(half_length, -half_width),
            pygame.Vector2(tip_x, 0.0),
            pygame.Vector2(half_length, half_width),
            pygame.Vector2(-half_length, half_width),
        )

        return min(
            self._distance_to_segment(
                point,
                polygon[i],
                polygon[(i + 1) % len(polygon)],
            )
            for i in range(len(polygon))
        )

    def detach_coeficient(
        self,
        host: MassObject,
        body_lengths: float,
        gravitational_constant: float,
        fixed_timestep: float,
        max_steps: int = 100_000,
    ) -> int | None:
        """Número de passos de empuxo contínuo para subir ``body_lengths``
        vezes o próprio comprimento a partir de ``host``.

        O foguete parte em repouso relativo à superfície, na altitude de
        acoplamento (``host.radius + body_length/2 + DOCKING_MARGIN``) e
        com o bico apontando radialmente para fora. A cada passo, o motor
        aplica ``thrust_acceleration`` no eixo radial (para fora) e a
        gravidade de ``host`` puxa na direção oposta. A posição radial é
        integrada em Euler semi-implícito — mesmo esquema do
        ``TrajectoryVisualizer`` — até que o afastamento atinja
        ``body_lengths * body_length``.

        Devolve ``None`` quando o empuxo é insuficiente para vencer a
        gravidade local em qualquer ponto da subida (``a_thrust <= g``),
        quando algum parâmetro é degenerado, ou quando ``max_steps`` é
        atingido sem convergir (teto de segurança contra configurações
        extremas).
        """
        if body_lengths <= 0.0 or fixed_timestep <= 0.0:
            return None
        if self.thrust_acceleration <= 0.0 or host.mass <= 0.0:
            return None

        radius = host.radius + self.body_length * 0.5 + self.DOCKING_MARGIN
        target = radius + body_lengths * self.body_length

        a_thrust = self.thrust_acceleration
        velocity = 0.0

        for step in range(1, max_steps + 1):
            gravity = gravitational_constant * host.mass / (radius * radius)
            acceleration = a_thrust - gravity
            if acceleration <= 0.0:
                return None
            velocity += acceleration * fixed_timestep
            radius += velocity * fixed_timestep
            if radius >= target:
                return step

        return None

    def attach_coeficient(
        self,
        host: "MassObject",
        gravitational_constant: float,
        landing_max_speed: float,
        fixed_timestep: float,
        max_nose_angle: float = math.radians(45.0),
        max_range_factor: float = 5.0,
        max_coast_steps: int = 5_000,
        max_burn_steps: int = 5_000,
    ) -> int | None:
        """Passos de empuxo de frenagem para pousar em segurança sobre ``host``.

        Modelo em duas fases:

        1. O foguete **coasta** (motor desligado) a partir do estado atual
           até o ponto onde a frenagem deve começar.
        2. Aplica retro-empuxo contínuo — bico apontado aproximadamente
           radial para fora — até cruzar a altitude de acoplamento.

        O número devolvido é o **tempo de queima** da fase 2, em passos. É
        o menor T que garante chegada com velocidade total dentro de
        ``landing_max_speed``: o ponto de corte é aquele em que a queima
        começa o mais tarde possível sem que a velocidade de chegada
        exceda o envelope. Queimar por mais tempo só aumentaria a
        velocidade de chegada.

        Pré-condições (devolvem ``None``):

        - bico a no máximo ``max_nose_angle`` da direção radial para fora;
        - ``r`` entre ``r_dock`` e ``max_range_factor × r_dock``;
        - velocidade radial negativa (aproximação);
        - componente tangencial dentro de ``landing_max_speed``;
        - parâmetros físicos válidos.

        Devolve ``None`` também quando a manobra é fisicamente impossível:
        mesmo começando a queimar o mais tarde possível, a velocidade de
        chegada excede ``landing_max_speed``.
        """
        if self.thrust_acceleration <= 0.0 or host.mass <= 0.0:
            return None
        if fixed_timestep <= 0.0 or landing_max_speed <= 0.0:
            return None

        r_dock = host.radius + self.body_length * 0.5 + self.DOCKING_MARGIN

        relative_position = self.position - host.position
        relative_velocity = self.velocity - host.velocity

        r = relative_position.length()
        if r <= r_dock or r > r_dock * max_range_factor:
            return None

        radial_hat = relative_position / r
        v_r = relative_velocity.dot(radial_hat)
        v_t = abs(relative_velocity.cross(radial_hat))

        if v_r >= 0.0:
            return None
        if v_t > landing_max_speed:
            return None

        nose = self.forward
        cos_alpha = nose.dot(radial_hat)
        if cos_alpha < math.cos(max_nose_angle):
            return None

        a_thrust = self.thrust_acceleration * cos_alpha
        if a_thrust <= 0.0:
            return None

        max_radial_speed_sq = landing_max_speed * landing_max_speed - v_t * v_t
        if max_radial_speed_sq <= 0.0:
            return None
        max_radial_speed = max_radial_speed_sq ** 0.5

        mu = gravitational_constant * host.mass

        # Coast até a janela de queima abrir. Em cada passo, testamos se um
        # burn contínuo a partir daqui chega à casca com velocidade segura.
        for _ in range(max_coast_steps):
            r_b = r
            v_r_b = v_r

            for burn_step in range(1, max_burn_steps + 1):
                a_r = a_thrust - mu / (r_b * r_b)
                v_r_b += a_r * fixed_timestep
                r_b += v_r_b * fixed_timestep

                if r_b <= r_dock:
                    if abs(v_r_b) <= max_radial_speed:
                        return burn_step
                    # Chegou rápido demais mesmo com burn máximo: coast
                    # adicional só agrava o quadro.
                    return None

                if v_r_b >= 0.0:
                    # Burn reverteu antes da casca — a janela ainda não
                    # abriu, coasta mais um passo e tenta de novo.
                    break
            else:
                # Nem chegou nem inverteu em ``max_burn_steps``: estado
                # patológico, desiste.
                return None

            # Coast um passo (apenas gravidade).
            v_r += (-mu / (r * r)) * fixed_timestep
            r += v_r * fixed_timestep

            if r <= r_dock or v_r >= 0.0:
                return None

        return None