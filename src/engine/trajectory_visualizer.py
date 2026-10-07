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

        Culling: corpos fora da câmera continuam sendo integrados — a
        gravidade deles afeta os demais — mas a trajetória deles não é
        acumulada nem desenhada, o que economiza uma transformação de
        tela e um par de floats por ponto por passo.

        Parada em colisão: a predição interrompe no primeiro passo em que
        qualquer par entra no raio de colisão. A simulação real funde os
        corpos nesse ponto, então prever adiante seria fisicamente inútil
        e visualmente confuso (as trajetórias se cruzariam).
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

        # Estado temporário em listas planas de floats.
        px = [b.position.x for b in bodies]
        py = [b.position.y for b in bodies]
        vx = [b.velocity.x for b in bodies]
        vy = [b.velocity.y for b in bodies]
        masses = [b.mass for b in bodies]
        radii = [b.radius for b in bodies]

        ax = [0.0] * count
        ay = [0.0] * count

        # Uma polilinha por corpo visível. Corpos fora da câmera ficam com
        # `None` e nunca acumulam pontos durante a predição.
        paths: list[list[tuple[float, float]] | None] = [None] * count
        for i in range(count):
            if camera.is_visible((px[i], py[i]), radii[i]):
                sp = camera.world_to_screen((px[i], py[i]))
                paths[i] = [(sp.x, sp.y)]

        for _ in range(steps):
            proximity = gravity.compute_accelerations_flat(
                px, py, masses, ax, ay,
                radii=radii,
            )
            if proximity:
                # Corpos prestes a colidir: a simulação real os fundirá.
                # Prever adiante produziria trajetórias que atravessam umas
                # às outras, sem valor físico nem visual.
                break

            for i in range(count):
                vx[i] += ax[i] * dt
                vy[i] += ay[i] * dt

                px[i] += vx[i] * dt
                py[i] += vy[i] * dt

                path = paths[i]
                if path is not None:
                    sp = camera.world_to_screen((px[i], py[i]))
                    path.append((sp.x, sp.y))

        for path in paths:
            if path is not None and len(path) >= 2:
                renderer.draw_polyline(self.color, path, width=self.width)

        return perf_counter() - start