"""
Mocked PromptServer for testing purposes in Impact Pack.
It'll display a simple Tkinter messagebox while mocked server is starting.
"""

import tkinter as tk
from tkinter import ttk
import utils as _  # noqa: F401


def show_loading(parent, message="Loading MockServer...") -> tk.Toplevel:
    """Show a loading window with a message."""
    win = tk.Toplevel(parent)
    win.title("Loading")
    win.geometry("300x80")
    win.transient(parent)
    win.grab_set()
    label = tk.Label(win, text=message, font=("TkDefaultFont", 12))
    label.pack(expand=True, fill="both", padx=20, pady=20)
    win.update()
    return win


# Mock PromptServer for Impact Pack
class MockServer:
    """
    A mocked PromptServer for testing purposes in Impact Pack.
    """

    def __init__(self):
        self.routes = self
        self.last_node_id = "mock_node_id"

    def post(self, route):
        def decorator(func):
            return func

        return decorator

    def get(self, route):
        def decorator(func):
            return func

        return decorator

    def add_on_prompt_handler(self, handler):
        pass

    def send_sync(self, event, data, sid=None):
        pass


root = tk.Tk()
style = ttk.Style()
if "clam" in style.theme_names():
    style.theme_use("clam")

# Mostrar ventana de carga
loading_win = show_loading(root)

import server  # noqa: E402 pylint: disable=C0413

# Aquí va la inicialización lenta
server.PromptServer.instance = MockServer()

# Cerrar ventana de carga
loading_win.destroy()
root.destroy()
