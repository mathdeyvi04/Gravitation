from collections.abc import Sequence
from src.entities.massobject import MassObject
import pygame

from src.physics.integrator import AccelerationModel


class CompositeAccelerationModel:
    """@brief Combina um modelo base com acelerações específicas dos corpos.

    @details O modelo base (ex.: `NewtonianGravity`) calcula a aceleração
    de todos os corpos. Em seguida, apenas os corpos que sobrescrevem
    `additional_acceleration` (como o `Rocket`) contribuem com um termo
    extra — tipicamente o empuxo.
    """

    __slots__ = (
        "base_model",
        "bodies",
        "_thrusters"
    )

    def __init__(
        self,
        base_model: AccelerationModel,
        bodies: Sequence,
    ) -> None:
        """@brief Configura o modelo base e pré-seleciona os thrusters.

        @details A lista `_thrusters` é montada uma única vez comparando
        a identidade do método `additional_acceleration` com o da classe
        base. Comparar funções de classe evita instanciar o método em
        cada passo e reduz o laço quente a apenas os corpos relevantes.

        @param base_model Modelo de aceleração que cobre todos os corpos.
        @param bodies Corpos da simulação, na mesma ordem usada pelo
            integrador.
        """
        self.base_model = base_model
        self.bodies = bodies
        default = MassObject.additional_acceleration
        # Corpos cujo `additional_acceleration` foi sobrescrito.
        # A comparação é por identidade de função de classe, não por
        # `isinstance`, para não pagar custo de MRO a cada passo.
        self._thrusters = [
            (i, body) for i, body in enumerate(bodies)
            if type(body).additional_acceleration is not default
        ]

    def compute_accelerations(
        self,
        positions: Sequence[pygame.Vector2],
        velocities: Sequence[pygame.Vector2],
        masses: Sequence[float],
    ) -> list[pygame.Vector2]:
        """@brief Aceleração total de cada corpo no estado dado.

        @details Delega ao modelo base e, para cada thruster, soma a
        aceleração específica do corpo (ex.: empuxo) à aceleração
        gravitacional já calculada.

        @param positions Posições atuais dos corpos.
        @param velocities Velocidades atuais dos corpos.
        @param masses Massas dos corpos.
        @return Lista de acelerações, na mesma ordem dos corpos.
        """
        accelerations = (
            self.base_model.compute_accelerations(
                positions,
                velocities,
                masses,
            )
        )

        # Só os corpos com `additional_acceleration` sobrescrito
        # entram aqui; os demais já estão cobertos pelo modelo base.
        for i, body in self._thrusters:
            extra = body.additional_acceleration(positions[i], velocities[i])
            if extra is not None:
                accelerations[i] += extra

        return accelerations