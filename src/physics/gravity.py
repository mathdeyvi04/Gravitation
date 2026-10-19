from collections.abc import Sequence

import pygame


class NewtonianGravity:
    """@brief Modelo de gravitação newtoniana para um sistema de N corpos.

    @details Trabalha com posições, velocidades e massas fornecidas
    externamente, permitindo que o integrador avalie o campo em estados
    intermediários sem tocar nos corpos reais.
    """

    __slots__ = (
        "gravitational_constant",
        "softening",
    )

    def __init__(
        self,
        gravitational_constant: float,
        softening: float = 0.0,
    ) -> None:
        """@brief Configura a constante gravitacional e o softening.

        @param gravitational_constant Valor de G.
        @param softening Suavização do denominador, em unidades de mundo.
        """
        self.gravitational_constant = gravitational_constant
        self.softening = softening

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        """@brief Calcula a aceleração de cada corpo no estado fornecido.

        @details `velocities` integra a interface para modelos futuros
        dependentes de velocidade; a gravidade atual não a utiliza.
        Aproveita a simetria do par (i, j) para computar a força uma
        única vez e aplicá-la com sinais opostos.

        @param positions Posições atuais dos corpos.
        @param velocities Velocidades atuais dos corpos.
        @param masses Massas dos corpos.
        @return Lista de acelerações, na mesma ordem dos corpos.
        """
        count = len(masses)

        accelerations = [
            pygame.Vector2(0.0, 0.0)
            for _ in range(count)
        ]

        eps_sq = self.softening * self.softening
        gravitational_constant = self.gravitational_constant

        for i in range(count):
            position_i = positions[i]
            mass_i = masses[i]

            # Apenas j > i: cada par é visitado uma vez e a força
            # resultante é distribuída simetricamente entre i e j.
            for j in range(i + 1, count):
                position_j = positions[j]
                mass_j = masses[j]

                delta = position_j - position_i

                # Softening entra somado ao quadrado da distância,
                # suavizando a força em aproximações extremas.
                distance_squared = delta.length_squared() + eps_sq

                if distance_squared < 1e-12:
                    continue

                # fator = G / r², mas com o produto escalar embutido:
                # |Δ|⁻¹ · |Δ|⁻² = |Δ|⁻³, e multiplicar por Δ já
                # entrega o vetor unitário escalado.
                inv_distance = distance_squared ** -0.5
                factor = gravitational_constant * inv_distance / distance_squared

                accelerations[i] += (
                    delta * (factor * mass_j)
                )

                accelerations[j] -= (
                    delta * (factor * mass_i)
                )

        return accelerations

    def compute_accelerations_flat(
            self,
            px: list[float],
            py: list[float],
            masses: list[float],
            ax: list[float],
            ay: list[float],
            radii: list[float] | None = None,
            overlaps: list[tuple[int, int]] | None = None,
            skip_pair: tuple[int, int] | None = None,
    ) -> None:
        """@brief Versão otimizada da gravidade para a previsão de trajetória.

        @details Trabalha diretamente com listas de floats para evitar
        criar `pygame.Vector2` no laço quente. A detecção de sobreposição
        é embutida no próprio laço de pares, evitando uma segunda
        varredura O(N²).

        @param px Coordenadas X.
        @param py Coordenadas Y.
        @param masses Massas dos corpos.
        @param ax Buffer de saída para as acelerações em X.
        @param ay Buffer de saída para as acelerações em Y.
        @param radii Raios dos corpos, necessários para detectar
            sobreposições.
        @param overlaps Lista onde os pares sobrepostos são anexados,
            se fornecida junto com `radii`.
        @param skip_pair Par `(i, j)` que nunca conta como sobreposto —
            usado para o foguete acoplado e seu hospedeiro.
        """
        count = len(masses)
        eps_sq = self.softening * self.softening
        gravitational_constant = self.gravitational_constant

        for i in range(count):
            ax[i] = 0.0
            ay[i] = 0.0

        detect = overlaps is not None and radii is not None
        skip_i = skip_pair[0] if skip_pair is not None else -1
        skip_j = skip_pair[1] if skip_pair is not None else -1

        for i in range(count):
            mass_i = masses[i]
            # Corpos absorvidos por fusão ficam com massa zero e são
            # ignorados no campo gravitacional.
            if mass_i <= 0.0:
                continue

            xi = px[i]
            yi = py[i]

            for j in range(i + 1, count):
                mass_j = masses[j]
                if mass_j <= 0.0:
                    continue

                dx = px[j] - xi
                dy = py[j] - yi
                raw_sq = dx * dx + dy * dy

                # A checagem de sobreposição usa a distância *sem*
                # softening: é contato físico, não gravidade.
                if detect and not (i == skip_i and j == skip_j):
                    threshold = radii[i] + radii[j]
                    if raw_sq <= threshold * threshold:
                        overlaps.append((i, j))

                distance_squared = raw_sq + eps_sq

                if distance_squared < 1e-12:
                    continue

                inv_distance = distance_squared ** -0.5
                factor = gravitational_constant * inv_distance / distance_squared

                # Fatores pré-calculados: força de i sobre j é
                # `factor · m_i`, e vice-versa. Evita multiplicar
                # massas dentro do acúmulo.
                force_scale_i = factor * mass_j
                force_scale_j = factor * mass_i

                ax[i] += dx * force_scale_i
                ay[i] += dy * force_scale_i

                ax[j] -= dx * force_scale_j
                ay[j] -= dy * force_scale_j