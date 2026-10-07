import math
from src.physics.gravity import NewtonianGravity
from time import perf_counter
from src.engine.camera import Camera2D
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
    )

    def __init__(
        self,
        steps: int = 600,
        color: tuple[int, int, int] = (170, 170, 255),
        width: int = 1,
    ) -> None:
        if steps < 1:
            raise ValueError("steps deve ser maior ou igual a 1.")

        if width < 1:
            raise ValueError("width deve ser maior ou igual a 1.")

        self.enabled = False
        self.steps = steps
        self.color = color
        self.width = width

    def toggle(self) -> None:
        """Alterna a visualização da trajetória futura."""
        self.enabled = not self.enabled

    def draw(
            self,
            renderer: Renderer,
            camera: Camera2D,
            bodies,
            gravity: NewtonianGravity,
            fixed_delta_time: float,
    ) -> float | None:
        """Prevê e desenha a trajetória futura dos corpos.

        Toda a previsão acontece em estado temporário. Nenhum objeto
        real da simulação é alterado. Devolve o tempo gasto na operação
        em segundos, ou `None` quando a visualização está desligada ou
        os parâmetros impedem a predição.
        """
        if not self.enabled:
            return None

        if not bodies:
            return None

        if fixed_delta_time <= 0.0:
            return None

        start = perf_counter()

        count = len(bodies)
        steps = self.steps
        dt = fixed_delta_time

        # Estado temporário em listas planas de floats. Evita alocações
        # de `pygame.Vector2` dentro do laço quente.
        px = [b.position.x for b in bodies]
        py = [b.position.y for b in bodies]
        vx = [b.velocity.x for b in bodies]
        vy = [b.velocity.y for b in bodies]
        masses = [b.mass for b in bodies]

        # Buffers de aceleração preenchidos pelo gravitador a cada passo.
        ax = [0.0] * count
        ay = [0.0] * count

        # Uma polilinha em screen_space por corpo. O primeiro ponto já
        # entra aqui para que a integração a seguir só precise anexar.
        paths: list[list[tuple[float, float]]] = []
        for i in range(count):
            sp = camera.world_to_screen((px[i], py[i]))
            paths.append([(sp.x, sp.y)])

        for _ in range(steps):
            gravity.compute_accelerations_flat(px, py, masses, ax, ay)

            for i in range(count):
                vx[i] += ax[i] * dt
                vy[i] += ay[i] * dt

                px[i] += vx[i] * dt
                py[i] += vy[i] * dt

                sp = camera.world_to_screen((px[i], py[i]))
                paths[i].append((sp.x, sp.y))

        for path in paths:
            renderer.draw_polyline(self.color, path, width=self.width)

        return perf_counter() - start