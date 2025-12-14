"""Mocked PromptServer for testing purposes in Impact Pack."""

import utils as _  # noqa: F401
import server


# Mock PromptServer for Impact Pack
class MockServer:
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


server.PromptServer.instance = MockServer()
