import pygame


class Input:
    """Estado de teclado e mouse a partir dos eventos do Pygame.

    Três categorias de estado, com semânticas distintas:

    - `_keys_down`: pressionadas agora; persiste entre frames.
    - `_keys_pressed`: pressionadas neste frame; limpo em `update`.
    - `_keys_released`: soltas neste frame; limpo em `update`.

    Chame `update()` uma vez por frame, antes de qualquer consulta.
    """

    # Distância máxima (ao quadrado, em pixels) entre pressionar e soltar
    # para que o gesto seja classificado como clique. Acima disso vira
    # arrasto — e o botão esquerdo passa a significar pan em vez de seleção.
    DRAG_THRESHOLD_SQ = 25  # ~5 px

    def __init__(self) -> None:
        """Inicializa os conjuntos de estado e zera o pedido de quit."""
        self._keys_down: set[int] = set()
        self._keys_pressed: set[int] = set()
        self._keys_released: set[int] = set()
        self._mouse_buttons_pressed: set[int] = set()
        self._mouse_buttons_down: set[int] = set()
        self._mouse_buttons_released: set[int] = set()
        self._mouse_buttons_clicked: set[int] = set()
        self._drag_start: dict[int, pygame.Vector2] = {}

        self.mouse_wheel = 0
        self.mouse_delta = pygame.Vector2(0.0, 0.0)
        self.mouse_position = pygame.Vector2(0.0, 0.0)
        self._events: list[pygame.event.Event] = []
        self.quit_requested = False

    def update(self) -> None:
        """Limpa o estado de um frame, drena eventos e atualiza o mouse.

        Deve ser chamada uma vez por iteração, antes de qualquer
        consulta. `quit_requested` não é resetado automaticamente.
        """
        previous_position = pygame.Vector2(self.mouse_position)

        self._keys_pressed.clear()
        self._keys_released.clear()
        self._mouse_buttons_pressed.clear()
        self._mouse_buttons_released.clear()
        self._mouse_buttons_clicked.clear()
        self.mouse_wheel = 0

        self.mouse_position.update(pygame.mouse.get_pos())
        self.mouse_delta = self.mouse_position - previous_position
        self._events = pygame.event.get()

        for event in self._events:
            if event.type == pygame.QUIT:
                self.quit_requested = True

            elif event.type == pygame.KEYDOWN:
                self._keys_down.add(event.key)
                self._keys_pressed.add(event.key)

            elif event.type == pygame.KEYUP:
                self._keys_down.discard(event.key)
                self._keys_released.add(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Botões 4/5 são a roda do mouse (scroll).
                if event.button == 4:
                    self.mouse_wheel += 1
                    continue
                if event.button == 5:
                    self.mouse_wheel -= 1
                    continue

                self._mouse_buttons_pressed.add(event.button)
                self._mouse_buttons_down.add(event.button)
                self._drag_start[event.button] = pygame.Vector2(event.pos)
                # A posição do evento é mais precisa que o poll do topo.
                self.mouse_position.update(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_buttons_down.discard(event.button)
                self._mouse_buttons_released.add(event.button)

                start = self._drag_start.pop(event.button, None)
                if start is not None:
                    end = pygame.Vector2(event.pos)
                    if (end - start).length_squared() <= self.DRAG_THRESHOLD_SQ:
                        self._mouse_buttons_clicked.add(event.button)

    def is_down(self, key: int) -> bool:
        """Indica se `key` está pressionada agora; persiste entre frames."""
        return key in self._keys_down

    def was_pressed(self, key: int) -> bool:
        """Indica se `key` foi pressionada neste frame."""
        return key in self._keys_pressed

    def was_released(self, key: int) -> bool:
        """Indica se `key` foi solta neste frame."""
        return key in self._keys_released

    def was_mouse_button_pressed(self, button: int) -> bool:
        """Indica se o botão `button` foi pressionado neste frame.

        Convenção do Pygame: 1 = esquerdo, 2 = meio, 3 = direito.
        """
        return button in self._mouse_buttons_pressed

    def is_mouse_button_down(self, button: int) -> bool:
        """Indica se `button` está pressionado agora; persiste entre frames."""
        return button in self._mouse_buttons_down

    def was_mouse_button_released(self, button: int) -> bool:
        """Indica se `button` foi solto neste frame."""
        return button in self._mouse_buttons_released

    def was_mouse_button_clicked(self, button: int) -> bool:
        """Indica se `button` foi pressionado e solto neste frame sem arrasto.

        Arrastos acima de `DRAG_THRESHOLD_SQ` são classificados como pan,
        não como clique, e por isso não disparam este predicado.
        """
        return button in self._mouse_buttons_clicked

    @property
    def events(self) -> list[pygame.event.Event]:
        """Eventos brutos do frame atual; não guarde entre frames."""
        return self._events