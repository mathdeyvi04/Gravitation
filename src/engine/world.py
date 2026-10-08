from collections.abc import Iterator
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity

from typing import TypeVar, Generic
T = TypeVar("T", bound=Entity)

class World(Generic[T]):
    """Contêiner genérico de entidades.

    Delega `update`, `fixed_update` e `render` a cada entidade e purga
    as inativas em uma passada única. Não contém física, colisão nem
    qualquer lógica específica de simulação — essas decisões vivem na
    cena concreta, que opera sobre `self.entities`.
    """

    __slots__ = ("entities",)

    def __init__(self) -> None:
        """Inicializa uma `World` vazia."""
        self.entities: list[T] = []

    def add(self, entity: T) -> T:
        """Adiciona `entity` e a devolve (permite encadeamento)."""
        self.entities.append(entity)
        return entity

    def extend(self, entities: list[T]) -> None:
        """Adiciona várias entidades de uma só vez."""
        self.entities.extend(entities)

    def remove(self, entity: T) -> None:
        """Remove `entity`, se presente. Silencioso se não estiver."""
        try:
            self.entities.remove(entity)
        except ValueError:
            pass

    def clear(self) -> None:
        """Esvazia a lista de entidades."""
        self.entities.clear()

    def __iter__(self) -> Iterator[T]:
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def __contains__(self, entity: T) -> bool:
        return entity in self.entities

    def update(self, delta_time: float) -> None:
        """Chama `update(delta_time)` em cada entidade."""
        for entity in self.entities:
            entity.update(delta_time)

    def fixed_update(self, fixed_delta_time: float) -> None:
        """Chama `fixed_update(fixed_delta_time)` em cada entidade."""
        for entity in self.entities:
            entity.fixed_update(fixed_delta_time)

    def visible_entities(
        self,
        camera: Camera2D,
        alpha: float = 1.0,
    ) -> Iterator[T]:
        """Itera sobre entidades ativas visíveis pela `camera`."""
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
        """Chama `render(renderer, camera, alpha)` em cada entidade."""
        for entity in self.entities:
            entity.render(renderer, camera, alpha)

    def purge_inactive(self) -> None:
        """Remove entidades com `active == False` em uma passada O(N)."""
        self.entities = [e for e in self.entities if e.active]