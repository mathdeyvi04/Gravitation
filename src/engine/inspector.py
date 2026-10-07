from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pygame

from src.entities.entity import Entity


@dataclass(frozen=True)
class InspectionProperty:
    """Propriedade exibível no inspector.

    `getter` extrai o valor de uma entidade; `formatter` o converte em
    string para exibição. Ambos recebem a entidade como argumento.
    """

    label: str
    getter: Callable[[Any], Any]
    formatter: Callable[[Any], str]


def format_vector(value: pygame.Vector2) -> str:
    """Formata um `Vector2` como `(x, y)` com 3 casas decimais."""
    return f"({value.x:.3f}, {value.y:.3f})"


class Inspector:
    """Consulta propriedades da entidade selecionada.

    Não conhece tipos concretos de entidade: o tipo-alvo e suas
    propriedades são registrados externamente via `register`. A
    resolução de propriedades percorre a MRO, então uma subclasse
    herda o registro da classe-mãe se não tiver o próprio.
    """

    def __init__(self) -> None:
        """Inicializa sem entidade selecionada e sem registros."""
        self._selected: Entity | None = None
        self._registry: dict[
            type[Entity],
            tuple[InspectionProperty, ...],
        ] = {}

    def register(
        self,
        entity_type: type[Entity],
        *properties: InspectionProperty,
    ) -> None:
        """Associa `properties` ao tipo `entity_type`.

        Sobrescreve o registro anterior do mesmo tipo, se houver.
        """
        self._registry[entity_type] = tuple(properties)

    def select(self, entity: Entity | None) -> None:
        """Define a entidade a ser inspecionada. `None` desseleciona."""
        self._selected = entity

    def clear(self) -> None:
        """Remove a seleção atual."""
        self._selected = None

    @property
    def selected(self) -> Entity | None:
        """Entidade selecionada, ou `None` se estiver inativa ou ausente.

        Uma entidade destruída (`active == False`) é tratada como
        desselecionada, sem que o chamador precise limpar o estado.
        """
        if self._selected is None or not self._selected.active:
            return None
        return self._selected

    def get_properties(self) -> tuple[InspectionProperty, ...]:
        """Propriedades registradas para o tipo da entidade selecionada.

        Retorna a primeira tupla encontrada ao percorrer a MRO do tipo,
        permitindo herança de registro. Retorna `()` se não há seleção
        ou nenhum ancestral foi registrado.
        """
        entity = self.selected
        if entity is None:
            return ()

        for cls in type(entity).__mro__:
            properties = self._registry.get(cls)
            if properties is not None:
                return properties

        return ()


class InspectorHUD:
    """Painel de inspeção desenhado no canto superior esquerdo.

    Lê a entidade selecionada e suas propriedades registradas pelo
    `Inspector` e as apresenta como um retângulo com título e uma
    linha por propriedade. O HUD não conhece tipos concretos de
    entidade — apenas consome o que o `Inspector` expõe.
    """

    def __init__(
        self,
        margin: int = 10,
        padding: int = 12,
        title_size: int = 24,
        property_size: int = 20,
        line_spacing: int = 6,
    ) -> None:
        """Configura as métricas de layout do painel.

        `margin` afasta o painel da borda da tela; `padding` afasta o
        conteúdo das bordas internas do painel. `title_size` e
        `property_size` são os tamanhos de fonte do título e das
        linhas, e `line_spacing` é o espaço vertical entre linhas
        consecutivas.
        """
        self.margin = margin
        self.padding = padding
        self.title_size = title_size
        self.property_size = property_size
        self.line_spacing = line_spacing

    def draw(
        self,
        renderer,
        inspector: Inspector,
    ) -> None:
        """Desenha o painel se houver entidade selecionada com propriedades.

        Não faz nada quando não há seleção ou quando o tipo da entidade
        não tem propriedades registradas. O layout é calculado a partir
        da largura da linha mais longa; a caixa cresce para acomodar o
        conteúdo em vez de ter tamanho fixo.
        """
        entity = inspector.selected

        if entity is None:
            return

        properties = inspector.get_properties()

        if not properties:
            return

        title = f"CORPO: {type(entity).__name__}"

        title_width, title_height = renderer.measure_text(
            title,
            size=self.title_size,
        )

        lines: list[str] = []

        max_width = title_width

        for property_ in properties:

            value = property_.getter(entity)
            formatted_value = property_.formatter(value)

            line = (
                f"{property_.label}: "
                f"{formatted_value}"
            )

            lines.append(line)

            width, _ = renderer.measure_text(
                line,
                size=self.property_size,
            )

            max_width = max(
                max_width,
                width,
            )

        line_height = renderer.measure_text(
            "Ag",
            size=self.property_size,
        )[1]

        height = (
            self.padding * 2
            + title_height
            + self.line_spacing
            + len(lines) * line_height
            + max(0, len(lines) - 1) * self.line_spacing
        )

        rect = pygame.Rect(
            self.margin,
            self.margin,
            max_width + self.padding * 2,
            height,
        )

        renderer.draw_rect(
            (10, 10, 15),
            rect,
        )

        renderer.draw_rect(
            (180, 180, 180),
            rect,
            width=1,
        )

        x = rect.x + self.padding
        y = rect.y + self.padding

        renderer.draw_text(
            title,
            (x, y),
            size=self.title_size,
            color=(255, 220, 120),
        )

        y += title_height + self.line_spacing

        for line in lines:

            renderer.draw_text(
                line,
                (x, y),
                size=self.property_size,
                color=(255, 255, 255),
            )

            y += line_height + self.line_spacing