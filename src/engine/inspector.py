from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pygame

from src.entities.entity import Entity

_TITLE_COLOR = (255, 255, 255)
_PRIMARY_TEXT_COLOR = (255, 255, 255)
_REFERENCE_TEXT_COLOR = (255, 190, 70)
_BACKGROUND_COLOR = (10, 10, 15)
_BORDER_COLOR = (180, 180, 180)

@dataclass(frozen=True)
class InspectionProperty:
    """Propriedade exibível no inspector.

    `getter` extrai o valor de uma entidade; `formatter` o converte em
    string para exibição. Ambos recebem a entidade como argumento.
    """

    label: str
    getter: Callable[[Any], Any]
    formatter: Callable[[Any], str]

@dataclass(frozen=True)
class RelativeInspectionProperty:
    """Propriedade calculada a partir de dois corpos."""

    label: str
    getter: Callable[[Any], Any]
    formatter: Callable[[Any], str]

class RelationInspector:
    """
    Mantém o corpo de referência e as propriedades relacionais.

    O corpo principal continua sendo responsabilidade do Inspector.
    """

    def __init__(self) -> None:
        self._reference: Entity | None = None

        self._registry: dict[
            tuple[type[Entity], type[Entity]],
            tuple[
                Callable[[Entity, Entity], Any],
                tuple[RelativeInspectionProperty, ...],
            ],
        ] = {}

    def register(
        self,
        primary_type: type[Entity],
        reference_type: type[Entity],
        context_factory: Callable[[Entity, Entity], Any],
        *properties: RelativeInspectionProperty,
    ) -> None:
        """Registra uma análise entre dois tipos de entidade."""

        self._registry[
            (primary_type, reference_type)
        ] = (
            context_factory,
            tuple(properties),
        )

    def select_reference(
        self,
        entity: Entity | None,
    ) -> None:
        self._reference = entity

    def clear(self) -> None:
        self._reference = None

    @property
    def reference(self) -> Entity | None:
        if (
            self._reference is None
            or not self._reference.active
        ):
            return None

        return self._reference

    def _get_registration(
        self,
        primary: Entity,
    ):
        reference = self.reference

        if reference is None:
            return None

        # Permite que subclasses herdem registros.
        for primary_class in type(primary).__mro__:
            for reference_class in type(reference).__mro__:

                registration = self._registry.get(
                    (
                        primary_class,
                        reference_class,
                    )
                )

                if registration is not None:
                    return registration

        return None

    def get_context(
        self,
        primary: Entity,
    ) -> Any | None:

        reference = self.reference

        if reference is None or reference is primary:
            return None

        registration = self._get_registration(primary)

        if registration is None:
            return None

        context_factory, _ = registration

        return context_factory(
            primary,
            reference,
        )

    def get_properties(
        self,
        primary: Entity,
    ) -> tuple[RelativeInspectionProperty, ...]:

        registration = self._get_registration(primary)

        if registration is None:
            return ()

        _, properties = registration

        return properties


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
        relation_inspector: RelationInspector | None = None,
    ) -> None:
        """Desenha o painel se houver entidade selecionada com propriedades.

        Não faz nada quando não há seleção ou quando o tipo da entidade
        não tem propriedades registradas. As propriedades do corpo
        principal saem em branco (mesma cor do anel de seleção); as que
        dependem do corpo de referência saem em laranja (mesma cor do
        anel de referência), incluindo o cabeçalho da seção.
        """
        entity = inspector.selected

        if entity is None:
            return

        properties = inspector.get_properties()

        if not properties:
            return

        relation_context = None
        relation_properties = ()
        reference = None

        if relation_inspector is not None:
            reference = relation_inspector.reference

            if reference is not None:
                relation_context = relation_inspector.get_context(entity)

                if relation_context is not None:
                    relation_properties = relation_inspector.get_properties(entity)

        title = f"CORPO: {type(entity).__name__}"

        title_width, title_height = renderer.measure_text(
            title,
            size=self.title_size,
        )

        primary_lines: list[str] = []
        max_width = title_width

        for property_ in properties:
            value = property_.getter(entity)
            formatted_value = property_.formatter(value)

            line = f"{property_.label}: {formatted_value}"

            primary_lines.append(line)

            width, _ = renderer.measure_text(
                line,
                size=self.property_size,
            )

            max_width = max(max_width, width)

        reference_lines: list[str] = []

        if (
                reference is not None
                and relation_context is not None
                and relation_properties
        ):
            reference_title = (
                f"REFERÊNCIA: {type(reference).__name__}"
            )

            reference_lines.append(reference_title)

            width, _ = renderer.measure_text(
                reference_title,
                size=self.property_size,
            )

            max_width = max(max_width, width)

            for property_ in relation_properties:
                value = property_.getter(relation_context)
                formatted_value = property_.formatter(value)

                line = f"{property_.label}: {formatted_value}"

                reference_lines.append(line)

                width, _ = renderer.measure_text(
                    line,
                    size=self.property_size,
                )

                max_width = max(max_width, width)

        total_lines = len(primary_lines) + len(reference_lines)

        line_height = renderer.measure_text(
            "Ag",
            size=self.property_size,
        )[1]

        height = (
                self.padding * 2
                + title_height
                + self.line_spacing
                + total_lines * line_height
                + max(0, total_lines - 1) * self.line_spacing
        )

        rect = pygame.Rect(
            self.margin,
            self.margin,
            max_width + self.padding * 2,
            height,
        )

        renderer.draw_rect(_BACKGROUND_COLOR, rect)
        renderer.draw_rect(_BORDER_COLOR, rect, width=1)

        x = rect.x + self.padding
        y = rect.y + self.padding

        renderer.draw_text(
            title,
            (x, y),
            size=self.title_size,
            color=_TITLE_COLOR,
        )

        y += title_height + self.line_spacing

        for line in primary_lines:
            renderer.draw_text(
                line,
                (x, y),
                size=self.property_size,
                color=_PRIMARY_TEXT_COLOR,
            )

            y += line_height + self.line_spacing

        for line in reference_lines:
            renderer.draw_text(
                line,
                (x, y),
                size=self.property_size,
                color=_REFERENCE_TEXT_COLOR,
            )

            y += line_height + self.line_spacing