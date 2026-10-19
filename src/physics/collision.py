import pygame

from src.entities.entity import Entity


def circles_overlap(
    center_a: pygame.Vector2,
    radius_a: float,
    center_b: pygame.Vector2,
    radius_b: float,
) -> bool:
    """@brief Indica se dois círculos em world_space se sobrepõem.

    @details Compara quadrados de distância com o quadrado da soma dos
    raios, evitando a raiz quadrada.

    @param center_a Centro do primeiro círculo.
    @param radius_a Raio do primeiro círculo.
    @param center_b Centro do segundo círculo.
    @param radius_b Raio do segundo círculo.
    @return `True` se os discos se interceptam ou se tocam.
    """
    r = radius_a + radius_b
    return center_a.distance_squared_to(center_b) <= r * r


def collides(
    a: Entity,
    b: Entity,
    alpha: float = 1.0,
) -> bool:
    """@brief Indica se as formas de colisão de `a` e `b` se sobrepõem.

    @details Cada forma é um conjunto de círculos; basta um par
    (um de `a`, um de `b`) se sobrepor para o teste ser verdadeiro.
    Entidades sem forma colidível (tupla vazia) nunca colidem.

    @param a Primeira entidade.
    @param b Segunda entidade.
    @param alpha Fator de interpolação entre o estado físico anterior
        e o atual, coerente com o que está sendo desenhado.
    @return `True` se houver sobreposição entre os círculos das formas.
    """
    # Produto cartesiano dos círculos das duas formas — retorno na
    # primeira colisão encontrada, sem montar pares intermediários.
    for center_a, radius_a in a.collision_circles(alpha):
        for center_b, radius_b in b.collision_circles(alpha):
            if circles_overlap(center_a, radius_a, center_b, radius_b):
                return True
    return False