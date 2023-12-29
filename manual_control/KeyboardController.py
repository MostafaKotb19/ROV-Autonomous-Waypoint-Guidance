import numpy as np
from pynput import keyboard

class KeyboardController:
    def __init__(self):
        """
        Initialize the KeyboardController.

        Initializes variables for pressed keys, force, and starts the keyboard listener.
        """
        self.pressed_keys = list()
        self.force = 100
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        """
        Handle key press events.

        Appends the pressed key to the list of pressed keys.

        Parameters:
        - key: The pressed key.
        """
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        """
        Handle key release events.

        Removes the released key from the list of pressed keys.

        Parameters:
        - key: The released key.
        """
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def parse_keys(self):
        """
        Parse pressed keys and generate control commands.

        Returns:
        - command: Control command generated based on pressed keys.
        """
        command = np.zeros(8)
        if 'i' in self.pressed_keys:
            command[0:4] += self.force
        if 'k' in self.pressed_keys:
            command[0:4] -= self.force
        if 'j' in self.pressed_keys:
            command[[4, 7]] += self.force
            command[[5, 6]] -= self.force
        if 'l' in self.pressed_keys:
            command[[4, 7]] -= self.force
            command[[5, 6]] += self.force

        if 'w' in self.pressed_keys:
            command[4:8] += self.force
        if 's' in self.pressed_keys:
            command[4:8] -= self.force
        if 'a' in self.pressed_keys:
            command[[4, 6]] += self.force
            command[[5, 7]] -= self.force
        if 'd' in self.pressed_keys:
            command[[4, 6]] -= self.force
            command[[5, 7]] += self.force

        return command