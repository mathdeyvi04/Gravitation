import math
from typing import Optional, Sequence
import torch
import pygame

from src.config import SimulationConfig
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.massobject import MassObject


class Rocket(MassObject):
    """@brief Corpo controlável com orientação, rotação e propulsão.

    @details Pode estar **acoplado** a um hospedeiro (`docked_to`).
    Enquanto acoplado, posição e velocidade acompanham o hospedeiro a
    cada passo fixo, e a orientação é fixada na direção radial — como se
    estivesse pousado. Qualquer comando do jogador (rotação ou empuxo)
    desacopla.
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
        "fuel",
        "fuel_max",
        "fuel_thrust_rate",
        "fuel_rotation_rate",
        # Gravado por `fixed_update` e lido no integrador: reflete se o
        # tanque tinha combustível no início do passo. Para rotação não
        # existe esquema equivalente — ela é aplicada direto no ângulo.
        "_thrust_authorized",
    )

    ROCKET_COLOR = (220, 220, 230)
    NOSE_COLOR = (240, 110, 80)

    # Folga entre a superfície do hospedeiro e a extremidade traseira
    # do foguete, em unidades de mundo. Evita que os círculos de
    # colisão se toquem tangencialmente logo no primeiro frame.
    DOCKING_MARGIN = 2.0

    # Índices de ação definidos em src/ai/DESIGN.md §3.
    AI_ACTIONS = (
        (-1, False),  # 0: girar à esquerda, motor desligado
        (0, False),   # 1: não girar, motor desligado
        (1, False),   # 2: girar à direita, motor desligado
        (-1, True),   # 3: girar à esquerda, motor ligado
        (0, True),    # 4: não girar, motor ligado
        (1, True),    # 5: girar à direita, motor ligado
    )

    def __init__(
        self,
        config: SimulationConfig,
        mass: float,
        position: pygame.Vector2,
        velocity: Optional[pygame.Vector2] = None,
        fuel_max: Optional[float] = None,
        fuel: Optional[float] = None,
        alpha: float = 2.5,
        angle: float = -math.pi * 0.5,
        body_length: float = 24.0,
        body_width: float = 10.0,
        nose_length: float = 12.0,
        rotation_speed: float = math.radians(120.0),
        thrust_acceleration: float = 120.0,
        color: tuple[int, int, int] = ROCKET_COLOR,
        nose_color: tuple[int, int, int] = NOSE_COLOR,
    ) -> None:
        """@brief Cria o foguete e o registra como `MassObject`.

        @details O raio de colisão herdado é o círculo envolvente do
        corpo + bico; a colisão real usa `collision_circles`, composta
        por quatro círculos alinhados ao eixo. `fuel_max` e `fuel` caem
        para os valores de `config` quando omitidos.

        @param config Configuração de simulação (fonte de defaults).
        @param mass Massa do foguete.
        @param position Posição inicial em world_space.
        @param velocity Velocidade inicial; zero se omitida.
        @param fuel_max Capacidade do tanque; usa `config.fuel_max` se None.
        @param fuel Carga inicial; usa `fuel_max` se None.
        @param alpha Parâmetro relativo à órbita alvo
        @param angle Orientação inicial, em radianos.
        @param body_length Comprimento do corpo (sem o bico).
        @param body_width Largura do corpo.
        @param nose_length Comprimento do bico.
        @param rotation_speed Velocidade angular de giro, em rad/s.
        @param thrust_acceleration Aceleração produzida pelo motor.
        @param color Cor do corpo.
        @param nose_color Cor do bico.
        """
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
        self.fuel_max = float(config.fuel_max if fuel_max is None else fuel_max)
        self.fuel = float(self.fuel_max if fuel is None else fuel)
        self.fuel_thrust_rate = float(config.fuel_thrust_rate)
        self.fuel_rotation_rate = float(config.fuel_rotation_rate)

        self._thrust_authorized = False

        self.body_length = float(body_length)
        self.body_width = float(body_width)
        self.nose_length = float(nose_length)

        self.nose_color = nose_color

        self.docked_to: Optional[MassObject] = None
        self.docking_angle: float = 0.0

        self.alpha = alpha

    # -- Acoplamento ---------------------------------------------------

    def dock_to(self, host: MassObject, angle: float) -> None:
        """@brief Acopla o foguete à superfície de `host`, no ângulo `angle`.

        @details `angle` é a direção radial (em world_space) do centro
        do hospedeiro até o foguete; o bico nasce apontando nessa mesma
        direção. A transição é tratada como colisão perfeitamente
        inelástica: o par passa a compartilhar a velocidade do centro
        de massa, preservando o momento linear do sistema.

        @param host Corpo ao qual o foguete será acoplado.
        @param angle Direção radial do centro até o foguete.
        """
        self.docked_to = host
        self.docking_angle = float(angle)
        self.angle = self.docking_angle
        self.previous_angle = self.angle

        # Colisão inelástica: ambos passam a compartilhar o centro de
        # massa. Sem isso, o resíduo `m_r · (v_r − v_h)` seria
        # descartado silenciosamente — invisível para planetas, mas
        # mensurável para luas e meteoros.
        total_mass = self.mass + host.mass
        if total_mass > 0.0:
            common_velocity = (
                                      self.velocity * self.mass + host.velocity * host.mass
                              ) / total_mass
            self.velocity.update(common_velocity)
            host.velocity.update(common_velocity)

        self._sync_to_host()

    def undock(self) -> None:
        """@brief Desacopla o foguete, deixando-o sob controle pleno."""
        self.docked_to = None

    def _sync_to_host(self) -> None:
        """@brief Ajusta posição e velocidade para o estado do hospedeiro.

        @details Não toca em `self.angle` — a orientação é controlada
        pelo jogador via `fixed_update`, mesmo enquanto acoplado. É o
        que permite girar o foguete sobre a superfície sem que ele caia.
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
        """@brief Re-sincroniza com o hospedeiro após o passo de integração.

        @details RK4 escreve em `position` e `velocity` uniformemente,
        sem saber do acoplamento. Reposicionamos aqui para que o foguete
        continue "grudado" na superfície. Se o hospedeiro foi destruído
        (por fusão), desacopla e segue livre.
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
        """@brief Direção do bico em world_space."""
        return pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle),
        )

    def set_control(
            self,
            rotation_input: int,
            thrusting: bool,
    ) -> None:
        """@brief Aplica comandos do jogador para este frame.

        @details Girar enquanto acoplado apenas muda a orientação do
        foguete — ele permanece pousado. Só o empuxo desacopla, e só se
        houver combustível. A borda de subida (`not self.thrusting`) é o
        gatilho: evitar re-desacoplar a cada frame de empuxo contínuo.

        @param rotation_input -1, 0 ou +1 (esquerda, nada, direita).
        @param thrusting Estado do motor para este frame.
        """
        thrusting = bool(thrusting)
        if (
            self.docked_to is not None
            and thrusting
            and not self.thrusting
            and self.fuel > 0.0
        ):
            # Sincroniza antes de soltar para herdar a velocidade do
            # hospedeiro — o foguete não "dispara" ao desacoplar.
            self._sync_to_host()
            self.undock()

        self.rotation_input = max(-1, min(1, int(rotation_input)))
        self.thrusting = thrusting

    def set_ai_action(self, action_index: int) -> None:
        """! @brief Aplica uma das seis ações discretas de `AI_ACTIONS`.

        @details Traduz o índice da política em `(rotation_input, thrusting)`
        e delega a `set_control`, reutilizando toda a lógica de desacoplamento,
        consumo de combustível e borda de subida do motor.

        @param action_index Índice inteiro em `[0, len(AI_ACTIONS))`.
        @throws ValueError Se `action_index` não for um inteiro válido
            (bool, float, string, fora do range).
        """
        # `bool` é subclasse de `int` em Python: `True == 1` passaria na
        # conversão e na checagem `action != action_index`, mapeando
        # silenciosamente para a ação 1. Rejeitamos antes de qualquer cast.
        if isinstance(action_index, bool):
            raise ValueError("O índice da ação deve ser um inteiro entre 0 e 5.")

        # `int()` cobre strings numéricas, floats, tensores escalares e
        # objetos com `__int__`; também levanta para NaN, `None` e afins.
        try:
            action = int(action_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("O índice da ação deve ser um inteiro entre 0 e 5.") from exc

        # Duas checagens em uma:
        # - `action != action_index` rejeita floats truncados (`3.5 → 3`),
        #   strings (`"3" → 3`, mas `"3" != 3`) e qualquer objeto cujo
        #   round-trip `int(x) == x` não seja exato.
        # - `0 <= action < len(...)` garante índice dentro da tabela.
        if action != action_index or not 0 <= action < len(self.AI_ACTIONS):
            raise ValueError("O índice da ação deve ser um inteiro entre 0 e 5.")

        rotation_input, thrusting = self.AI_ACTIONS[action]
        self.set_control(rotation_input, thrusting)

    def control_with_network(
        self,
        network,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> int:
        """@brief Consulta uma rede `ActorCritic` e aplica a ação escolhida.

        @details O vetor `observation` segue o `DESIGN.md §5` (9 valores). A
        construção desse vetor é responsabilidade do ambiente de treino —
        esta função só valida a forma, encaminha à rede e repassa o índice
        escolhido para `set_ai_action`.

        @param network Instância com `select_action` e (opcionalmente)
            `n_actions` e `obs_dim`.
        @param observation Observação escalar ou em lote de tamanho 1.
        @param deterministic Se `True`, a política escolhe a ação de maior
            probabilidade; se `False`, amostra da distribuição.
        @return Índice inteiro da ação efetivamente aplicada ao foguete.
        @throws TypeError Se `network` não expõe `select_action`.
        @throws ValueError Se `network` não puder ser usada (sem `n_actions`,
            sem parâmetros, forma de observação incompatível, saída ambígua).
        """

        if not hasattr(network, "select_action"):
            raise TypeError("A rede deve disponibilizar select_action().")

        # Validação "lenient": se `n_actions` não existe, assume o tamanho
        # de `AI_ACTIONS`. Isso permite redes duck-typed simples em testes
        # sem forçar um contrato de atributo. Redes reais devem declarar.
        if getattr(network, "n_actions", len(self.AI_ACTIONS)) != len(self.AI_ACTIONS):
            raise ValueError("A rede deve produzir exatamente seis ações.")

        # `next(iter(params()))` resolve o device de forma barata e sem
        # assumir `nn.Module`. Três falhas possíveis, todas convertidas:
        # - AttributeError: não tem `.parameters()`;
        # - StopIteration: `.parameters()` está vazio (rede sem pesos);
        # - TypeError: `.parameters()` não é iterável.
        try:
            device = next(network.parameters()).device
        except (AttributeError, StopIteration, TypeError) as exc:
            raise ValueError("A rede precisa ter parâmetros PyTorch.") from exc

        # `as_tensor` só copia se dtype/device divergirem; caso contrário,
        # compartilha memória com `observation`. Mantemos `float32` porque é
        # o dtype canônico do `ActorCritic`.
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        )
        # Aceitamos observação 1-D `(obs_dim,)` e adicionamos a dimensão de
        # lote, pois `select_action` espera `(batch, obs_dim)`.
        if observation_tensor.ndim == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        # Mesma validação "lenient": sem `obs_dim`, a forma passada é o
        # próprio contrato (útil em smoke tests). Com `obs_dim`, a checagem
        # de forma é estrita e inclui a dimensão de lote == 1.
        expected_dim = getattr(network, "obs_dim", observation_tensor.shape[-1])
        if observation_tensor.shape != (1, expected_dim):
            raise ValueError(
                f"Esperada observação com forma ({expected_dim},) ou "
                f"(1, {expected_dim}); recebida {tuple(observation_tensor.shape)}."
            )

        # `no_grad`: inferência pura — sem grafo, sem alocação de ativações
        # intermediárias. A tupla devolvida é `(ações, log_prob, valor)`.
        with torch.no_grad():
            actions, _log_prob, _value = network.select_action(
                observation_tensor, deterministic=deterministic
            )
        # Defensivo: se a rede devolver um lote (ex.: `deterministic=False`
        # amostrando múltiplas vezes), abortamos em vez de escolher `[0]`
        # silenciosamente — o chamador perceberá o contrato violado.
        if actions.numel() != 1:
            raise ValueError("control_with_network espera uma única observação por chamada.")

        # `.item()` extrai o escalar Python do tensor (device-agnóstico).
        action = int(actions.item())
        self.set_ai_action(action)
        return action

    def try_land(
        self,
        bodies: Sequence[MassObject],
        max_relative_speed: float,
    ) -> bool:
        """@brief Acopla o foguete a um corpo ao cruzar a altitude de acoplagem.

        @details O gatilho é a casca `r_dock = host.radius +
        body_length/2 + DOCKING_MARGIN`, alinhada com `_sync_to_host` e
        com `attach_coeficient` — não `collides`, que dispara duas
        unidades *abaixo* e desperdiçaria o envelope de velocidade.
        Corpos a alta velocidade são ignorados: o foguete pode passar
        raspando e ainda pousar em outro.

        @param bodies Corpos candidatos a hospedeiro.
        @param max_relative_speed Velocidade relativa máxima para pouso.
        @return `True` se acoplou a algum corpo nesta chamada.
        """
        if self.docked_to is not None:
            return False

        # Termo constante de `r_dock`; só `body.radius` muda entre corpos.
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

            # Teste em componentes: evita um `Vector2` temporário para
            # a diferença de velocidades no laço quente.
            dvx = self.velocity.x - body.velocity.x
            dvy = self.velocity.y - body.velocity.y
            if dvx * dvx + dvy * dvy > max_speed_sq:
                continue

            self.dock_to(body, math.atan2(delta.y, delta.x))
            return True

        return False

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Atualiza orientação e debita combustível em passo fixo.

        @details Roda tanto acoplado quanto livre — o jogador precisa
        mirar o bico antes de decolar; a posição, quando acoplado, é
        restaurada em `post_integrate`. O combustível é debitado **uma
        única vez por passo**, aqui — `additional_acceleration` roda
        quatro vezes por passo no RK4 e não deve tocar no tanque. O
        flag `_thrust_authorized` é gravado **antes** do débito para
        que o último passo de queima autorizado produza efeito completo.

        @param fixed_delta_time Passo fixo em segundos.
        """
        self.previous_angle = self.angle

        if self.fuel <= 0.0:
            self._thrust_authorized = False
            return

        self._thrust_authorized = self.thrusting

        if self.rotation_input != 0:
            self.angle += (
                    self.rotation_input
                    * self.rotation_speed
                    * fixed_delta_time
            )
            # Normaliza para (-π, π] — o ângulo cresce indefinidamente
            # ao girar sempre no mesmo sentido, e `sin`/`cos` não
            # precisam disso.
            self.angle = math.atan2(
                math.sin(self.angle),
                math.cos(self.angle),
            )
            self.fuel = max(
                0.0,
                self.fuel - self.fuel_rotation_rate * fixed_delta_time,
            )

        if self.thrusting:
            self.fuel = max(
                0.0,
                self.fuel - self.fuel_thrust_rate * fixed_delta_time,
            )

    def additional_acceleration(
        self,
        position: pygame.Vector2,
        velocity: pygame.Vector2,
    ) -> Optional[pygame.Vector2]:
        """@brief Aceleração produzida pelo motor no passo atual.

        @details Lê `_thrust_authorized` (estado do tanque no início do
        passo) em vez de `fuel` — quando o débito do passo zera o
        tanque, o empuxo já autorizado ainda precisa produzir efeito.

        @param position Posição avaliada pelo integrador (não usada).
        @param velocity Velocidade avaliada pelo integrador (não usada).
        @return Aceleração de empuxo, ou `None` se o motor não atua.
        """
        if self.docked_to is not None:
            return None
        if not self._thrust_authorized:
            return None
        if self.thrust_acceleration <= 0.0:
            return None
        return self.forward * self.thrust_acceleration

    def prediction_acceleration(
            self,
            elapsed_time: float,
    ) -> Optional[pygame.Vector2]:
        """@brief Aceleração do motor durante uma previsão futura.

        @details Não debita combustível — simula a trajetória como se o
        tanque durasse o horizonte. Tanque já vazio, porém, não mostra
        empuxo. O caso intermediário (acaba no meio da predição) é
        limitação conhecida.

        @param elapsed_time Tempo desde o início da predição.
        @return Aceleração de empuxo prevista, ou `None` se inativa.
        """
        if self.docked_to is not None:
            return None
        if self.fuel <= 0.0:
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
        """@brief Quatro círculos alinhados ao eixo do foguete.

        @details Todos têm raio igual à meia-largura do corpo, então
        tangenciam as laterais do retângulo. Os quatro centros cobrem a
        extensão total (traseira, meio, frente, bico) sem deixar vão.

        @param alpha Fator de interpolação entre os estados anterior e
            atual, coerente com o render.
        @return Tupla de `(centro, raio)` em world_space.
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

        # Os centros locais recuam `radius` das extremidades para que
        # os círculos não se projetem além do retângulo.
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
        """@brief Ângulo interpolado entre os estados anterior e atual.

        @details O delta é normalizado para (-π, π] antes da
        interpolação, de modo que o foguete sempre gira pelo caminho
        mais curto — inclusive ao cruzar a descontinuidade em ±π.

        @param alpha Fator de interpolação em [0, 1].
        @return Ângulo interpolado em radianos.
        """
        if alpha >= 1.0:
            return self.angle

        # `+π % 2π − π` traz o delta para (-π, π] sem branches.
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
        """@brief Vértices do retângulo do corpo em world_space.

        @param position Centro do foguete.
        @param angle Orientação em radianos.
        @return Lista com os 4 vértices do corpo, em world_space.
        """
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
        """@brief Vértices do triângulo do bico em world_space.

        @param position Centro do foguete.
        @param angle Orientação em radianos.
        @return Lista com os 3 vértices do bico, em world_space.
        """
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
        """@brief Contorno completo (corpo + bico) em world_space.

        @details Usado para `get_world_bounds` e `click_distance`; o
        `render` desenha corpo e bico como polígonos separados apenas
        para colorir o bico.

        @param position Centro do foguete.
        @param angle Orientação em radianos.
        @return Lista com os 5 vértices do contorno, em world_space.
        """
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
        """@brief Desenha corpo e bico do foguete.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação entre os estados anterior e
            atual.
        """
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
        """@brief AABB do foguete em world_space na posição interpolada.

        @param alpha Fator de interpolação em [0, 1].
        @return `(min_x, min_y, max_x, max_y)`.
        """
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
        """@brief Converte um ponto de world_space para o referencial local.

        @details Rotação inversa ao redor da posição interpolada do
        foguete; o resultado é o ponto no referencial em que o bico
        aponta para +x.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return Ponto no referencial local do foguete.
        """
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
        """@brief Indica se um ponto está dentro de um triângulo.

        @details Usa o sinal dos produtos vetoriais das arestas: o
        ponto está dentro quando não há mistura de sinais (todos os
        três do mesmo lado, ou exatamente zero).

        @param point Ponto a testar.
        @param a Primeiro vértice.
        @param b Segundo vértice.
        @param c Terceiro vértice.
        @return `True` se o ponto está dentro (borda inclusive).
        """
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
        """@brief Distância de um ponto a um segmento.

        @details Projeta o ponto sobre a reta e satura o parâmetro em
        [0, 1], de modo que pontos "além" das pontas medem a distância
        até o vértice correspondente.

        @param point Ponto a medir.
        @param a Início do segmento.
        @param b Fim do segmento.
        @return Distância em unidades de mundo.
        """
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
        """@brief Indica se o ponto pertence visualmente ao foguete.

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return `True` se o ponto cai no corpo ou no bico.
        """
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
        """@brief Distância do ponto ao contorno do foguete (0 se dentro).

        @param world_position Ponto em world_space.
        @param alpha Fator de interpolação em [0, 1].
        @return `0.0` se o ponto está dentro; menor distância ao
            contorno caso contrário.
        """
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
        max_steps: int = 100,
    ) -> int | None:
        """@brief Passos de empuxo contínuo para subir `body_lengths` corpos.

        @details Parte em repouso relativo à superfície, na altitude de
        acoplagem, com o bico radial para fora. Integra em Euler
        semi-implícito até o afastamento atingir
        `body_lengths * body_length`.

        @param host Corpo de referência.
        @param body_lengths Altura-alvo, em múltiplos de `body_length`.
        @param gravitational_constant Constante gravitacional.
        @param fixed_timestep Passo fixo em segundos.
        @param max_steps Teto de segurança.
        @return Passos de queima, ou `None` se a subida é impossível
            (`a_thrust <= g`) ou degenerada.
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
            # Como `radius` cresce a cada passo, `gravity` diminui —
            # se `a_thrust <= g` já falha no primeiro passo, nunca
            # haverá decolagem.
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
        max_coast_steps: int = 100,
        max_burn_steps: int = 100,
    ) -> int | None:
        """@brief Passos de frenagem para pousar em segurança sobre `host`.

        @details Modelo em duas fases: (1) coasta com o motor desligado
        até o ponto onde a frenagem deve começar; (2) aplica retro-empuxo
        contínuo até cruzar a altitude de acoplagem. O retorno é o menor
        tempo de queima da fase 2 que garante chegada dentro de
        `landing_max_speed`. Devolve `None` quando as pré-condições
        geométricas/dinâmicas falham ou quando a manobra é fisicamente
        impossível.

        @param host Corpo de destino.
        @param gravitational_constant Constante gravitacional.
        @param landing_max_speed Velocidade total máxima na casca.
        @param fixed_timestep Passo fixo em segundos.
        @param max_nose_angle Ângulo máximo entre bico e radial para fora.
        @param max_range_factor Alcance máximo em múltiplos de `r_dock`.
        @param max_coast_steps Teto de passos na fase de coast.
        @param max_burn_steps Teto de passos na fase de queima.
        @return Passos de queima, ou `None` se a manobra não é possível.
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

        # Só a componente radial do empuxo freia a queda; a tangencial
        # teria de ser cancelada por rotação e está fora do envelope.
        a_thrust = self.thrust_acceleration * cos_alpha
        if a_thrust <= 0.0:
            return None

        # Folga de velocidade disponível para a componente radial
        # depois de acomodar a tangencial.
        max_radial_speed_sq = landing_max_speed * landing_max_speed - v_t * v_t
        if max_radial_speed_sq <= 0.0:
            return None
        max_radial_speed = max_radial_speed_sq ** 0.5

        mu = gravitational_constant * host.mass

        # Estratégia: adiar o início da queima o máximo possível. Em
        # cada passo de coast, testamos a hipótese "queima a partir
        # daqui chega dentro do envelope?" — se sim, é o menor T.
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
                    # Chegou rápido demais com burn máximo: coastar
                    # mais só agrava o quadro.
                    return None

                if v_r_b >= 0.0:
                    # Burn reverteu antes da casca — a janela ainda
                    # não abriu; coasta um passo e tenta de novo.
                    break
            else:
                # Nem chegou nem inverteu em `max_burn_steps`: estado
                # patológico.
                return None

            # Coast um passo (apenas gravidade).
            v_r += (-mu / (r * r)) * fixed_timestep
            r += v_r * fixed_timestep

            if r <= r_dock or v_r >= 0.0:
                return None

        return None