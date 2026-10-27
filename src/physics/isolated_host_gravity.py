"""@brief Gravidade isolada entre um hospedeiro e múltiplos foguetes.

@details O hospedeiro (índice `host_index` na lista de corpos) é
tratado como referência fixa: sua aceleração é sempre zero,
independentemente da massa dos foguetes ao redor. Cada foguete sente
apenas a atração do hospedeiro — foguetes não se atraem entre si.

Esse modelo é o pano de fundo físico dos experimentos multi-foguete
do `Experimenter` (ver `train.py`). Ele isola os indivíduos de uma
população de redes neurais que compartilham a mesma janela.
"""

from collections.abc import Sequence

import pygame


class IsolatedHostGravity:
    """@brief Gravidade `host → foguetes` com hospedeiro estático."""

    __slots__ = ("gravitational_constant", "softening", "host_index")

    def __init__(
        self,
        gravitational_constant: float,
        softening: float = 0.0,
        host_index: int = 0,
    ) -> None:
        self.gravitational_constant = gravitational_constant
        self.softening = softening
        self.host_index = host_index

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        count = len(masses)
        accelerations = [pygame.Vector2(0.0, 0.0) for _ in range(count)]

        if count <= 1:
            return accelerations

        host_idx = self.host_index
        if host_idx < 0 or host_idx >= count:
            return accelerations

        host_mass = masses[host_idx]
        if host_mass <= 0.0:
            return accelerations

        host_pos = positions[host_idx]
        G = self.gravitational_constant
        eps_sq = self.softening * self.softening

        for i in range(count):
            if i == host_idx or masses[i] <= 0.0:
                continue

            delta = host_pos - positions[i]
            dist_sq = delta.length_squared() + eps_sq
            if dist_sq < 1e-12:
                continue

            inv_dist = dist_sq ** -0.5
            factor = G * host_mass * inv_dist / dist_sq
            accelerations[i] = delta * factor

        return accelerations