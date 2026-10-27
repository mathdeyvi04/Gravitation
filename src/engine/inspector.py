from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union
from inspect import getmro
import pygame

from src.entities.entity import Entity
from src.entities.massobject import MassObject
from src.entities.rocket import Rocket

AnyGrav = Union[Entity, MassObject, Rocket]

_TITLE_COLOR = (255, 255, 255)
_PRIMARY_TEXT_COLOR = (255, 255, 255)
_REFERENCE_TEXT_COLOR = (255, 190, 70)
_BACKGROUND_COLOR = (10, 10, 15)
_BORDER_COLOR = (180, 180, 180)

@dataclass(frozen=True)
class InspectionProperty:
    """@brief Propriedade exibível no inspector.

    @details `getter` extrai o valor de uma entidade; `formatter` o
    converte em string para exibição. Ambos recebem a entidade como
    argumento.

    `widget`, quando presente, é um callable
    `(entidade, renderer, rect) -> None` invocado logo abaixo da linha
    de texto da propriedade. Recebe um `pygame.Rect` com altura fixa
    (`InspectorHUD.WIDGET_HEIGHT`) onde desenhar o que quiser — útil
    para representações visuais que não cabem em texto, como um
    diagrama orientado do corpo selecionado.
    """

    label: str
    getter: Callable[[Any], Any]
    formatter: Callable[[Any], str]
    widget: Optional[Callable[[Any, Any, pygame.Rect], None]] = None
    primary: Optional[AnyGrav] = None
    reference: Optional[AnyGrav] = None

class RelationInspector:
    """@brief Mantém o corpo de referência e as propriedades relacionais.

    @details O corpo principal continua sendo responsabilidade do
    `Inspector`.
    """

    def __init__(self) -> None:
        """@brief Inicializa sem referência e sem registros."""
        self._reference: AnyGrav | None = None

        self._registry: dict[
            tuple[type, type],
            tuple[
                Callable[[Any, Any], Any],
                tuple[InspectionProperty, ...],
            ],
        ] = {}

    def register(
        self,
        primary_type: type[AnyGrav],
        reference_type: type[AnyGrav],
        context_factory: Callable[[Any, Any], Any],
        *properties: InspectionProperty,
    ) -> None:
        """@brief Registra uma análise entre dois tipos de entidade.

        @details `context_factory` é chamada sob demanda com
        `(primary, reference)`; o resultado é passado aos getters
        das propriedades relacionais.

        @param primary_type Tipo do corpo principal.
        @param reference_type Tipo do corpo de referência.
        @param context_factory Fábrica do contexto compartilhado.
        @param properties Propriedades relacionais exibíveis.
        """
        self._registry[
            (primary_type, reference_type)
        ] = (
            context_factory,
            tuple(properties),
        )

    def select_reference(
        self,
        entity: AnyGrav | None,
    ) -> None:
        """@brief Define (ou limpa) o corpo de referência.

        @param entity Corpo de referência; `None` desliga a análise
            relacional.
        """
        self._reference = entity

    def clear(self) -> None:
        """@brief Remove a referência atual."""
        self._reference = None

    @property
    def reference(self) -> AnyGrav | None:
        """@brief Referência ativa, ou `None` se ausente/inativa.

        @details Uma referência destruída é tratada como ausente, sem
        que o chamador precise limpar o estado.

        @return A referência ou `None`.
        """
        if (
            self._reference is None
            or not self._reference.active
        ):
            return None

        return self._reference

    def _get_registration(
        self,
        primary: AnyGrav,
    ):
        """@brief Resolve o registro aplicável ao par de tipos via MRO.

        @details Percorre a MRO de ambos os tipos (principal e
        referência) e devolve o primeiro registro encontrado — o que
        permite que subclasses herdem o registro da classe-mãe.

        @param primary Corpo principal.
        @return `(context_factory, properties)` ou `None`.
        """
        reference = self.reference

        if reference is None:
            return None

        # Permite que subclasses herdem registros.
        for primary_class in getmro(type(primary)):
            for reference_class in getmro(type(reference)):
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
        primary: AnyGrav,
    ) -> Any | None:
        """@brief Constrói o contexto compartilhado para o par atual.

        @details Devolve `None` se não há referência, se a referência
        é o próprio corpo principal, ou se não há registro aplicável.

        @param primary Corpo principal.
        @return Contexto produzido por `context_factory`, ou `None`.
        """
        reference = self.reference

        # Sem referência distinta do principal, não há relação a exibir.
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
        primary: AnyGrav,
    ) -> tuple[InspectionProperty, ...]:
        """@brief Propriedades relacionais registradas para o par atual.

        @param primary Corpo principal.
        @return Tupla de propriedades; `()` se não houver registro.
        """
        registration = self._get_registration(primary)

        if registration is None:
            return ()

        _, properties = registration

        return properties


class Inspector:
    """@brief Consulta propriedades da entidade selecionada.

    @details Não conhece tipos concretos de entidade: o tipo-alvo e
    suas propriedades são registrados externamente via `register`. A
    resolução de propriedades percorre a MRO, então uma subclasse herda
    o registro da classe-mãe se não tiver o próprio.
    """

    def __init__(self) -> None:
        """@brief Inicializa sem entidade selecionada e sem registros."""
        self._selected: AnyGrav | None = None
        self._registry: dict[
            type,
            tuple[InspectionProperty, ...],
        ] = {}

    def register(
        self,
        entity_type: type[AnyGrav],
        *properties: InspectionProperty,
    ) -> None:
        """@brief Associa `properties` ao tipo `entity_type`.

        @details Sobrescreve o registro anterior do mesmo tipo, se
        houver.

        @param entity_type Tipo registrado.
        @param properties Propriedades exibíveis para esse tipo.
        """
        self._registry[entity_type] = tuple(properties)

    def select(self, entity: AnyGrav | None) -> None:
        """@brief Define a entidade a ser inspecionada.

        @param entity Entidade a inspecionar; `None` desseleciona.
        """
        self._selected = entity

    def clear(self) -> None:
        """@brief Remove a seleção atual."""
        self._selected = None

    @property
    def selected(self) -> AnyGrav | None:
        """@brief Entidade selecionada, ou `None` se inativa/ausente.

        @details Uma entidade destruída (`active == False`) é tratada
        como desselecionada, sem que o chamador precise limpar o
        estado.

        @return A entidade selecionada ou `None`.
        """
        if self._selected is None or not self._selected.active:
            return None
        return self._selected

    def get_properties(self) -> tuple[InspectionProperty, ...]:
        """@brief Propriedades registradas para o tipo selecionado.

        @details Retorna a primeira tupla encontrada ao percorrer a MRO
        do tipo, permitindo herança de registro.

        @return Tupla de propriedades; `()` se não houver seleção ou
            registro.
        """
        entity = self.selected
        if entity is None:
            return ()

        for cls in getmro(type(entity)):
            properties = self._registry.get(cls)
            if properties is not None:
                return properties

        return ()


class InspectorHUD:
    """@brief Painel de inspeção desenhado no canto superior esquerdo.

    @details Lê a entidade selecionada e suas propriedades registradas
    pelo `Inspector` e as apresenta como um retângulo com título e uma
    linha por propriedade. O HUD não conhece tipos concretos de
    entidade — apenas consome o que o `Inspector` expõe.
    """

    # Altura reservada, em pixels, para widgets anexados a uma linha
    # de propriedade. Escolhido para caber um diagrama do foguete com
    # folga dentro do painel.
    WIDGET_HEIGHT = 56

    def __init__(
        self,
        margin: int = 10,
        padding: int = 12,
        title_size: int = 24,
        property_size: int = 20,
        line_spacing: int = 6,
    ) -> None:
        """@brief Configura as métricas de layout do painel.

        @details `margin` afasta o painel da borda da tela; `padding`
        afasta o conteúdo das bordas internas. `title_size` e
        `property_size` são os tamanhos de fonte do título e das linhas;
        `line_spacing` é o espaço vertical entre linhas consecutivas.

        @param margin Distância até a borda da tela, em pixels.
        @param padding Distância entre borda do painel e conteúdo.
        @param title_size Tamanho da fonte do título.
        @param property_size Tamanho da fonte das linhas.
        @param line_spacing Espaço vertical entre linhas.
        """
        self.margin = margin
        self.padding = padding
        self.title_size = title_size
        self.property_size = property_size
        self.line_spacing = line_spacing
        self._cached_width = 0
        self._last_entity_id: int | None = None
        self._last_reference_id: int | None = None

    @staticmethod
    def _measure_into(
        renderer,
        text: str,
        size: int,
        current_max: int,
    ) -> int:
        """@brief Mede `text` e devolve o máximo com `current_max`.

        @param renderer Renderizador usado para medir.
        @param text Texto a medir.
        @param size Tamanho da fonte.
        @param current_max Máximo acumulado até agora.
        @return `max(current_max, largura de text)`.
        """
        width, _ = renderer.measure_text(text, size=size)
        return width if width > current_max else current_max

    def draw(
        self,
        renderer,
        inspector: Inspector,
        relation_inspector: RelationInspector | None = None,
    ) -> None:
        """@brief Desenha o painel de inspeção, se houver seleção.

        @details Não faz nada quando não há seleção ou quando o tipo da
        entidade não tem propriedades registradas. Propriedades do corpo
        principal saem em branco; as que dependem da referência saem em
        laranja, incluindo o cabeçalho da seção.

        A largura do painel é estabilizada por um cache monotônico: o
        conteúdo muda a cada frame e recalcular a largura a partir do
        texto atual faria o painel tremer. A chave do cache é
        `(id(entidade), id(referência))` — troca de par descarta o
        cache; mesma seleção, a largura só cresce.

        @param renderer Renderizador alvo.
        @param inspector Inspector com a seleção atual.
        @param relation_inspector Inspector relacional opcional.
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

        # -- Estabilização da largura ----------------------------------
        # A chave de cache é o par (entidade, referência). Quando
        # qualquer um dos dois muda, o cache é descartado: o painel
        # parte do zero e cresce até caber o conteúdo atual. Enquanto
        # a chave for a mesma, `_cached_width` funciona como piso — a
        # largura só cresce, nunca encolhe, e os números que oscilam
        # no texto não propagam para o retângulo.
        entity_id = id(entity)
        reference_id = id(reference) if reference is not None else None

        if (
                entity_id != self._last_entity_id
                or reference_id != self._last_reference_id
        ):
            self._cached_width = 0
            self._last_entity_id = entity_id
            self._last_reference_id = reference_id

        title = type(entity).__name__

        title_width, title_height = renderer.measure_text(
            title,
            size=self.title_size,
        )

        primary_lines: list[str] = []
        primary_widgets: dict[int, Callable] = {}
        max_width = title_width

        for property_ in properties:
            value = property_.getter(entity)
            formatted_value = property_.formatter(value)

            line = f"{property_.label}: {formatted_value}"

            primary_lines.append(line)

            # A chave é o índice da linha já formatada — o widget é
            # desenhado logo abaixo dela.
            if property_.widget is not None:
                primary_widgets[len(primary_lines) - 1] = property_.widget

            max_width = self._measure_into(
                renderer,
                line,
                self.property_size,
                max_width,
            )

        reference_lines: list[str] = []

        if (
                reference is not None
                and relation_context is not None
                and relation_properties
        ):
            reference_title = (
                type(reference).__name__
            )

            # O cabeçalho entra como primeira linha; a cor laranja
            # aplica-se a toda a seção na hora de desenhar.
            reference_lines.append(reference_title)

            max_width = self._measure_into(
                renderer,
                reference_title,
                self.property_size,
                max_width,
            )

            for property_ in relation_properties:
                value = property_.getter(relation_context)
                formatted_value = property_.formatter(value)

                line = f"{property_.label}: {formatted_value}"

                reference_lines.append(line)

                max_width = self._measure_into(
                    renderer,
                    line,
                    self.property_size,
                    max_width,
                )

        # Aplica o piso e atualiza o cache. A ordem importa: primeiro o
        # piso é imposto à largura deste frame, depois o cache é
        # atualizado com o resultado — assim `_cached_width` reflete
        # exatamente o que foi desenhado e nunca decresce dentro de uma
        # mesma seleção.
        if max_width < self._cached_width:
            max_width = self._cached_width
        else:
            self._cached_width = max_width

        total_lines = len(primary_lines) + len(reference_lines)
        widget_count = len(primary_widgets)

        # Altura de linha via "Ag": aproxima o limite superior das
        # letras e evita depender do tamanho em pontos da fonte.
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
                + widget_count * (self.WIDGET_HEIGHT + self.line_spacing)
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

        for idx, line in enumerate(primary_lines):
            renderer.draw_text(
                line,
                (x, y),
                size=self.property_size,
                color=_PRIMARY_TEXT_COLOR,
            )

            y += line_height + self.line_spacing

            # Widget, se houver, ocupa a faixa logo abaixo da linha.
            widget = primary_widgets.get(idx)
            if widget is not None:
                widget_rect = pygame.Rect(
                    x, y, max_width, self.WIDGET_HEIGHT
                )
                widget(entity, renderer, widget_rect)
                y += self.WIDGET_HEIGHT + self.line_spacing

        for line in reference_lines:
            renderer.draw_text(
                line,
                (x, y),
                size=self.property_size,
                color=_REFERENCE_TEXT_COLOR,
            )

            y += line_height + self.line_spacing