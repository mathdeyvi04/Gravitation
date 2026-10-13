import pygame
from src.physics.gravity import NewtonianGravity
from time import perf_counter
from src.engine.camera import Camera2D
from src.entities.massobject import MassObject
from src.engine.renderer import Renderer


class TrajectoryVisualizer:
    """
    Calcula e desenha uma previsão da trajetória futura dos corpos.

    A previsão parte do estado atual de cada corpo e simula apenas
    a interação gravitacional entre eles por uma quantidade limitada
    de passos fixos.

    Os estados reais da simulação nunca são modificados.

    Toda a predição é feita em listas planas de floats (sem
    `pygame.Vector2` no laço interno), e cada trajetória é desenhada
    como uma única polilinha, reduzindo drasticamente o número de
    chamadas ao pygame por frame.
    """

    __slots__ = (
        "enabled",
        "steps",
        "color",
        "width",
        "pixel_step"
    )

    def __init__(
        self,
        steps: int = 600,
        color: tuple[int, int, int] = (170, 170, 255),
        width: int = 1,
        pixel_step: int =6
    ) -> None:
        if steps < 1:
            raise ValueError("steps deve ser maior ou igual a 1.")

        if width < 1:
            raise ValueError("width deve ser maior ou igual a 1.")

        self.enabled = False
        self.steps = steps
        self.color = color
        self.width = width

        self.pixel_step = pixel_step

    def toggle(self) -> None:
        """Alterna a visualização da trajetória futura."""
        self.enabled = not self.enabled

    @staticmethod
    def _merge_pair(
        px: list[float],
        py: list[float],
        vx: list[float],
        vy: list[float],
        masses: list[float],
        radii: list[float],
        i: int,
        j: int,
    ) -> None:
        """Funde o par (i, j) in-place, se ambos ainda estiverem vivos.

        Sobrevive o mais massivo, replicando a política do
        `_resolve_merges` real. Pares cujo lado já foi absorvido por
        outro merge no mesmo passo viram no-op — cadeias se resolvem
        ao longo dos passos da predição.
        """
        mass_i = masses[i]
        mass_j = masses[j]
        if mass_i <= 0.0 or mass_j <= 0.0:
            return

        if mass_j > mass_i:
            i, j = j, i
            mass_i, mass_j = mass_j, mass_i

        total = mass_i + mass_j

        px[i] = (px[i] * mass_i + px[j] * mass_j) / total
        py[i] = (py[i] * mass_i + py[j] * mass_j) / total
        vx[i] = (vx[i] * mass_i + vx[j] * mass_j) / total
        vy[i] = (vy[i] * mass_i + vy[j] * mass_j) / total
        radii[i] = (radii[i] ** 3 + radii[j] ** 3) ** (1.0 / 3.0)
        masses[i] = total

        masses[j] = 0.0
        vx[j] = 0.0
        vy[j] = 0.0
        radii[j] = 0.0

    from src.entities.massobject import MassObject  # no topo do arquivo

    def draw(
        self,
        renderer: Renderer,
        camera: Camera2D,
        bodies,
        gravity: NewtonianGravity,
        fixed_delta_time: float,
        focus_entities=None,
    ) -> float | None:
        """Prevê e desenha a trajetória futura dos corpos.

        Colisões não interrompem a predição: pares sobrepostos são
        fundidos in-place (massa, momento e centro de massa
        conservados), com o absorvido zerado. A detecção acontece
        dentro do próprio laço de gravidade, então cada passo faz uma
        única varredura O(N²).

        O foguete acoplado segue o hospedeiro em offset fixo, sem
        participar da detecção de sobreposição com ele.

        Desempenho:
        - **Stride adaptativo**: registra apenas ~2 pontos por pixel
          de tela disponível. Para uma janela 1500×1000, são ~5000
          pontos mesmo com `steps` muito maior. Os segmentos retos
          entre amostras percorrem o mesmo caminho que os pontos
          pulados — a curva desenhada é idêntica para qualquer
          trajetória razoavelmente suave.
        - **Pré-alocação**: cada lista de trajetória nasce com
          capacidade exata, evitando as realocações dobradas de
          `append`.
        - **Thrusters deduzidos uma vez por frame** via comparação
          de method de classe: corpos que não sobrescrevem
          `prediction_acceleration` (isto é, mundo menos o
          foguete) nem chegam a ser consultados no laço.
        """
        focus_entities = tuple(focus_entities or ())

        if not self.enabled and not focus_entities:
            return None
        if not bodies:
            return None
        if fixed_delta_time <= 0.0:
            return None
        if pygame.event.peek(pygame.QUIT):
            return None

        start = perf_counter()

        count = len(bodies)
        steps = self.steps
        dt = fixed_delta_time

        px = [b.position.x for b in bodies]
        py = [b.position.y for b in bodies]
        vx = [b.velocity.x for b in bodies]
        vy = [b.velocity.y for b in bodies]
        masses = [b.mass for b in bodies]
        radii = [b.radius for b in bodies]

        ax = [0.0] * count
        ay = [0.0] * count

        # Acoplamento: no máximo um foguete, no máximo um hospedeiro.
        # Guardamos os índices e o offset uma única vez — o offset é
        # constante em world_space porque o hospedeiro não gira.
        docked_body_idx = -1
        docked_host_idx = -1
        docked_offset_x = 0.0
        docked_offset_y = 0.0

        for i, body in enumerate(bodies):
            host = getattr(body, "docked_to", None)
            if host is None:
                continue
            for k, other in enumerate(bodies):
                if other is host:
                    docked_body_idx = i
                    docked_host_idx = k
                    docked_offset_x = body.position.x - host.position.x
                    docked_offset_y = body.position.y - host.position.y
                    break
            break  # só existe um foguete

        skip_pair = None
        if docked_body_idx >= 0:
            lo, hi = (
                (docked_body_idx, docked_host_idx)
                if docked_body_idx < docked_host_idx
                else (docked_host_idx, docked_body_idx)
            )
            skip_pair = (lo, hi)

        focus_ids = {id(entity) for entity in focus_entities}

        # -- Stride adaptativo -------------------------------------
        # `self.pixel_step` é o espaçamento, em pixels de tela, entre
        # amostras consecutivas da trajetória. Valores maiores = menos
        # pontos registrados = mais economia; menores = curva mais fiel
        # em zoom alto. O denominador usa o perímetro do viewport como
        # proxy da "extensão" da tela — uma janela maior comporta mais
        # amostras naturalmente.
        perimeter = camera.viewport_size.x + camera.viewport_size.y
        target_points = max(1, int(perimeter / self.pixel_step))
        stride = max(1, steps // target_points)

        # Capacidade exata da lista de cada trajetória. O `+2` cobre
        # a posição inicial e o último passo (forçado mesmo se não
        # cair no stride).
        capacity = (steps + stride - 1) // stride + 2

        cam_x = camera.position.x
        cam_y = camera.position.y
        cam_zoom = camera.zoom
        center_x = camera.viewport_center.x
        center_y = camera.viewport_center.y

        paths: list[list | None] = [None] * count
        # Número de pontos já escritos em cada lista de trajetória.
        # `lengths[i] == 0` significa que o corpo não está sendo
        # rastreado (fora da câmera e não focado).
        lengths = [0] * count

        for i, body in enumerate(bodies):
            should_draw = self.enabled or id(body) in focus_ids
            focused = id(body) in focus_ids
            # Focados desenham mesmo fora da câmera: o jogador precisa
            # ver para onde o corpo selecionado vai, mesmo quando ele
            # acabou de sair do quadro.
            if should_draw and (focused or camera.is_visible((px[i], py[i]), radii[i])):
                path = [None] * capacity
                path[0] = (
                    (px[i] - cam_x) * cam_zoom + center_x,
                    (py[i] - cam_y) * cam_zoom + center_y,
                )
                paths[i] = path
                lengths[i] = 1

        # Thrusters: corpos cujo `prediction_acceleration` foi
        # sobrescrito. Comparar funções de classe evita instanciar
        # o method em cada chamada — só o foguete entra aqui, e o
        # laço interno nem executa quando a lista está vazia.
        default_prediction = MassObject.prediction_acceleration
        thrusters = [
            (i, body) for i, body in enumerate(bodies)
            if type(body).prediction_acceleration is not default_prediction
        ]

        # Lista reutilizada pelo laço inteiro — sem realocação por passo.
        overlaps: list[tuple[int, int]] = []

        for step in range(steps):

            overlaps.clear()
            gravity.compute_accelerations_flat(
                px, py, masses, ax, ay,
                radii=radii,
                overlaps=overlaps,
                skip_pair=skip_pair,
            )

            if thrusters:
                prediction_time = (step + 1) * dt
                for i, body in thrusters:
                    if masses[i] <= 0.0:
                        continue
                    extra = body.prediction_acceleration(prediction_time)
                    if extra is not None:
                        ax[i] += extra.x
                        ay[i] += extra.y

            for i in range(count):
                if masses[i] <= 0.0:
                    continue
                vx[i] += ax[i] * dt
                vy[i] += ay[i] * dt
                px[i] += vx[i] * dt
                py[i] += vy[i] * dt

            for i, j in overlaps:
                self._merge_pair(px, py, vx, vy, masses, radii, i, j)

            if (
                    docked_body_idx >= 0
                    and masses[docked_body_idx] > 0.0
                    and masses[docked_host_idx] > 0.0
            ):
                k = docked_host_idx
                i = docked_body_idx
                px[i] = px[k] + docked_offset_x
                py[i] = py[k] + docked_offset_y
                vx[i] = vx[k]
                vy[i] = vy[k]

            # Só registra amostras nos múltiplos de `stride` — e
            # sempre no último passo, para fechar a curva.
            if step % stride == 0 or step == steps - 1:
                for i in range(count):
                    path = paths[i]
                    if path is None or masses[i] <= 0.0:
                        continue
                    path[lengths[i]] = (
                        (px[i] - cam_x) * cam_zoom + center_x,
                        (py[i] - cam_y) * cam_zoom + center_y,
                    )
                    lengths[i] += 1

        for i, path in enumerate(paths):
            if path is None:
                continue
            n = lengths[i]
            if n >= 2:
                renderer.draw_polyline(self.color, path[:n], width=self.width)

        return perf_counter() - start