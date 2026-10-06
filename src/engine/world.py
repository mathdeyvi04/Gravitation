from collections.abc import Iterator
from src.engine.camera import Camera2D
from src.engine.renderer import Renderer
from src.entities.entity import Entity

from typing import TypeVar, Generic
T = TypeVar("T", bound=Entity)

class World(Generic[T]):
    """Contêiner genérico de entidades.

    Responsabilidades:

    - Manter a lista de entidades e expor operações estáveis sobre ela.
    - Delegar `update`, `fixed_update` e `render` a cada entidade.
    - Purgar entidades inativas em uma única passada.

    A `World` **não** contém física, regras de colisão ou qualquer
    lógica específica de uma simulação. Essas decisões pertencem à
    cena concreta, que opera sobre as entidades expostas por
    `self.entities`. Isso mantém esta classe reaproveitável em outros
    projetos que usem `Entity` como base.

    A lista subjacente é `list[Entity]`. Índices espaciais ou outras
    estruturas (grade, quadtree) podem ser adicionados aqui no futuro
    sem alterar a API pública.
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
        """Remove entidades com `active == False` em uma passada O(N).

        A reconstrução via list comprehension roda em C e substitui a
        lista de uma só vez, sem invalidar iteradores que já tenham
        terminado (esta chamada é sempre o último passo de um frame).
        """
        self.entities = [e for e in self.entities if e.active]