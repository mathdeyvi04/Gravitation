from collections.abc import Iterator
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity

from typing import TypeVar, Generic
T = TypeVar("T", bound=Entity)

class World(Generic[T]):
    """@brief Contêiner genérico de entidades.

    @details Delega `update`, `fixed_update` e `render` a cada entidade
    e purga as inativas em uma passada única. Não contém física, colisão
    nem qualquer lógica específica de simulação — essas decisões vivem
    na cena concreta, que opera sobre `self.entities`.
    """

    __slots__ = ("entities",)

    def __init__(self) -> None:
        """@brief Inicializa uma `World` vazia."""
        self.entities: list[T] = []

    def add(self, entity: T) -> T:
        """@brief Adiciona `entity` e a devolve.

        @details O retorno permite encadeamento (ex.:
        `world.add(Rocket(...))`).

        @param entity Entidade a adicionar.
        @return A própria `entity`.
        """
        self.entities.append(entity)
        return entity

    def extend(self, entities: list[T]) -> None:
        """@brief Adiciona várias entidades de uma só vez.

        @param entities Entidades a adicionar.
        """
        self.entities.extend(entities)

    def remove(self, entity: T) -> None:
        """@brief Remove `entity`, se presente.

        @details Silencioso se `entity` não estiver na lista; a ausência
        não é erro para o chamador.

        @param entity Entidade a remover.
        """
        try:
            self.entities.remove(entity)
        except ValueError:
            pass

    def clear(self) -> None:
        """@brief Esvazia a lista de entidades."""
        self.entities.clear()

    def __iter__(self) -> Iterator[T]:
        """@brief Itera sobre as entidades na ordem de inserção.

        @return Iterador sobre `self.entities`.
        """
        return iter(self.entities)

    def __len__(self) -> int:
        """@brief Número de entidades no mundo.

        @return Tamanho de `self.entities`.
        """
        return len(self.entities)

    def __contains__(self, entity: T) -> bool:
        """@brief Indica se `entity` pertence ao mundo.

        @param entity Entidade a procurar.
        @return `True` se presente em `self.entities`.
        """
        return entity in self.entities

    def update(self, delta_time: float) -> None:
        """@brief Chama `update(delta_time)` em cada entidade.

        @param delta_time Tempo desde o último frame, em segundos.
        """
        for entity in self.entities:
            entity.update(delta_time)

    def fixed_update(self, fixed_delta_time: float) -> None:
        """@brief Chama `fixed_update(fixed_delta_time)` em cada entidade.

        @param fixed_delta_time Passo fixo em segundos.
        """
        for entity in self.entities:
            entity.fixed_update(fixed_delta_time)

    def visible_entities(
        self,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> Iterator[T]:
        """@brief Itera sobre entidades ativas visíveis pela `camera`.

        @details Corpos inativos são pulados antes de `is_visible`,
        evitando o custo de recorte para entidades que não seriam
        desenhadas de qualquer forma.

        @param camera Câmera com o viewport atual.
        @param alpha Fator de interpolação em [0, 1].
        @return Iterador das entidades ativas visíveis.
        """
        for entity in self.entities:
            if not entity.active:
                continue
            if entity.is_visible(camera, alpha):
                yield entity

    def render(
        self,
        renderer: Renderer,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> None:
        """@brief Chama `render(renderer, camera, alpha)` em cada entidade.

        @details A ordem de desenho segue a ordem de inserção: a última
        entidade adicionada é desenhada por cima das anteriores.

        @param renderer Renderizador alvo.
        @param camera Câmera usada para world_space → screen_space.
        @param alpha Fator de interpolação em [0, 1].
        """
        for entity in self.entities:
            entity.render(renderer, camera, alpha)

    def purge_inactive(self) -> None:
        """@brief Remove entidades com `active == False` em uma passada O(N).

        @details Substitui a lista por uma compreensão filtrada; os
        iteradores abertos sobre a lista antiga continuam válidos, mas
        novas consultas já veem apenas as ativas.
        """
        self.entities = [e for e in self.entities if e.active]