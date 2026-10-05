import pygame

class Input:
    def __init__(self):
        self._keys_down = set()
        self._keys_pressed = set()
        self._keys_released = set()

        self._events = []

        self.quit_requested = False

    def update(self) -> None:
        self._keys_pressed.clear()
        self._keys_released.clear()

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

    def is_down(self, key: int) -> bool:
        return key in self._keys_down

    def was_pressed(self, key: int) -> bool:
        return key in self._keys_pressed

    def was_released(self, key: int) -> bool:
        return key in self._keys_released

    @property
    def events(self):
        return self._events