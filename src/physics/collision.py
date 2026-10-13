import pygame

from src.entities.entity import Entity


def circles_overlap(
    center_a: pygame.Vector2,
    radius_a: float,
    center_b: pygame.Vector2,
    radius_b: float,
) -> bool:
    """Indica se dois círculos em world_space se sobrepõem.

    Compara quadrados de distância com o quadrado da soma dos raios —
    mesma otimização que `MassObject.overlaps` fazia.
    """
    r = radius_a + radius_b
    return center_a.distance_squared_to(center_b) <= r * r


def collides(
    a: Entity,
    b: Entity,
    alpha: float = 1.0,
) -> bool:
    """Indica se as formas de colisão de `a` e `b` se sobrepõem.

    Cada forma é um conjunto de círculos. A sobreposição ocorre se
    qualquer par (um de `a`, um de `b`) se sobrepõe. Entidades sem
    forma colidível (tupla vazia) nunca colidem.
    """
    for center_a, radius_a in a.collision_circles(alpha):
        for center_b, radius_b in b.collision_circles(alpha):
            if circles_overlap(center_a, radius_a, center_b, radius_b):
                return True
    return False