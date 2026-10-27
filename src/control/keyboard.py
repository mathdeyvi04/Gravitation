import pygame


class Input:
    """@brief Estado de teclado e mouse a partir dos eventos do Pygame.

    @details Três categorias de estado, com semânticas distintas:

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
        """@brief Inicializa os conjuntos de estado e zera o pedido de quit."""
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
        """@brief Limpa o estado de um frame, drena eventos e atualiza o mouse.

        @details Deve ser chamada uma vez por iteração, antes de
        qualquer consulta. `quit_requested` não é resetado
        automaticamente — uma vez `True`, permanece `True` até o
        chamador encerrar.

        O `mouse_delta` é calculado por diferença entre o poll do topo
        e a posição anterior; eventos `MOUSEBUTTONDOWN/UP` sobrescrevem
        `mouse_position` com a posição do evento, que é mais precisa que
        o poll.
        """
        previous_position = pygame.Vector2(self.mouse_position)

        # Estado "deste frame" é limpo antes de processar os eventos.
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
                # Botões 4/5 são a roda do mouse (scroll). O Pygame
                # não expõe `MOUSEWHEEL` em todas as versões/plataformas,
                # então tratamos pelo `MOUSEBUTTONDOWN`.
                if event.button == 4:
                    self.mouse_wheel += 1
                    continue
                if event.button == 5:
                    self.mouse_wheel -= 1
                    continue

                self._mouse_buttons_pressed.add(event.button)
                self._mouse_buttons_down.add(event.button)
                # Guarda o ponto de partida para decidir clique vs arrasto
                # no `MOUSEBUTTONUP`.
                self._drag_start[event.button] = pygame.Vector2(event.pos)
                # A posição do evento é mais precisa que o poll do topo.
                self.mouse_position.update(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._mouse_buttons_down.discard(event.button)
                self._mouse_buttons_released.add(event.button)

                start = self._drag_start.pop(event.button, None)
                if start is not None:
                    end = pygame.Vector2(event.pos)
                    # Deslocamento ao quadrado: comparação sem sqrt,
                    # coerente com `DRAG_THRESHOLD_SQ`.
                    if (end - start).length_squared() <= self.DRAG_THRESHOLD_SQ:
                        self._mouse_buttons_clicked.add(event.button)

    def is_down(self, key: int) -> bool:
        """@brief Indica se `key` está pressionada agora.

        @details Estado persistente entre frames.

        @param key Código de tecla do Pygame (ex.: `pygame.K_SPACE`).
        @return `True` se a tecla está atualmente pressionada.
        """
        return key in self._keys_down

    def was_pressed(self, key: int) -> bool:
        """@brief Indica se `key` foi pressionada neste frame.

        @param key Código de tecla do Pygame.
        @return `True` se houve `KEYDOWN` de `key` neste frame.
        """
        return key in self._keys_pressed

    def was_released(self, key: int) -> bool:
        """@brief Indica se `key` foi solta neste frame.

        @param key Código de tecla do Pygame.
        @return `True` se houve `KEYUP` de `key` neste frame.
        """
        return key in self._keys_released

    def was_mouse_button_pressed(self, button: int) -> bool:
        """@brief Indica se o botão `button` foi pressionado neste frame.

        @details Convenção do Pygame: 1 = esquerdo, 2 = meio,
        3 = direito.

        @param button Código de botão do Pygame.
        @return `True` se houve `MOUSEBUTTONDOWN` neste frame.
        """
        return button in self._mouse_buttons_pressed

    def is_mouse_button_down(self, button: int) -> bool:
        """@brief Indica se `button` está pressionado agora.

        @details Estado persistente entre frames.

        @param button Código de botão do Pygame.
        @return `True` se o botão está atualmente pressionado.
        """
        return button in self._mouse_buttons_down

    def was_mouse_button_released(self, button: int) -> bool:
        """@brief Indica se `button` foi solto neste frame.

        @param button Código de botão do Pygame.
        @return `True` se houve `MOUSEBUTTONUP` neste frame.
        """
        return button in self._mouse_buttons_released

    def was_mouse_button_clicked(self, button: int) -> bool:
        """@brief Indica se `button` foi pressionado e solto sem arrasto.

        @details Arrastos acima de `DRAG_THRESHOLD_SQ` são
        classificados como pan, não como clique, e por isso não
        disparam este predicado. O clique é registrado no
        `MOUSEBUTTONUP`, com base no deslocamento desde o
        `MOUSEBUTTONDOWN` correspondente.

        @param button Código de botão do Pygame.
        @return `True` se houve clique neste frame.
        """
        return button in self._mouse_buttons_clicked

    @property
    def events(self) -> list[pygame.event.Event]:
        """@brief Eventos brutos do frame atual.

        @details A lista é substituída em cada `update()`; não guarde
        referências entre frames.

        @return Lista de `pygame.event.Event` deste frame.
        """
        return self._events