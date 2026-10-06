import math

import pygame

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
        gravitational_constant: float,
        softening: float,
        fixed_delta_time: float,
    ) -> None:
        """
        Prevê e desenha a trajetória futura dos corpos.

        Toda a previsão acontece em estado temporário. Nenhum objeto
        real da simulação é alterado.
        """
        if not self.enabled:
            return

        if not bodies:
            return

        if fixed_delta_time <= 0.0:
            return

        count = len(bodies)
        steps = self.steps
        dt = fixed_delta_time
        G = gravitational_constant
        eps_sq = softening * softening

        # Estado temporário em listas planas de floats. Evita alocações
        # de `pygame.Vector2` dentro do laço quente.
        px = [b.position.x for b in bodies]
        py = [b.position.y for b in bodies]
        vx = [b.velocity.x for b in bodies]
        vy = [b.velocity.y for b in bodies]
        masses = [b.mass for b in bodies]

        # Pré-calcula dt/m por corpo. Zero para massless.
        dt_over_m = [
            (dt / m if m > 0.0 else 0.0)
            for m in masses
        ]

        # Uma polilinha em screen_space por corpo. O primeiro ponto
        # já entra aqui.
        paths: list[list[tuple[float, float]]] = []
        for i in range(count):
            sp = camera.world_to_screen((px[i], py[i]))
            paths.append([(sp.x, sp.y)])

        for _ in range(steps):

            # 1. Gravidade e impulso numa única passada.
            #    Cada par (i, j), i < j, é visitado exatamente uma vez
            #    e contribui para ambos os corpos. Elimina a lista
            #    `forces` e uma segunda iteração.
            for i in range(count):
                xi = px[i]
                yi = py[i]
                mi = masses[i]
                fxi = 0.0
                fyi = 0.0
                for j in range(i + 1, count):
                    dx = px[j] - xi
                    dy = py[j] - yi
                    dist_sq = dx * dx + dy * dy + eps_sq
                    if dist_sq < 1e-12:
                        continue
                    inv_dist = 1.0 / math.sqrt(dist_sq)
                    scale = G * mi * masses[j] * inv_dist / dist_sq
                    fxi += dx * scale
                    fyi += dy * scale
                    # Reação em j aplicada imediatamente.
                    vx[j] -= dx * scale * dt_over_m[j]
                    vy[j] -= dy * scale * dt_over_m[j]

                vx[i] += fxi * dt_over_m[i]
                vy[i] += fyi * dt_over_m[i]

            # 2. Integração semi-implícita + coleta do ponto de tela.
            for i in range(count):
                px[i] += vx[i] * dt
                py[i] += vy[i] * dt
                sp = camera.world_to_screen((px[i], py[i]))
                paths[i].append((sp.x, sp.y))

        # 3. Uma chamada de polilinha por corpo em vez de `steps`
        #    chamadas de segmento.
        for path in paths:
            renderer.draw_polyline(self.color, path, width=self.width)