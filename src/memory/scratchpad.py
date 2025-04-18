import asyncio
import json
from typing import Any

from rich.console import Console

console = Console()


class Scratchpad:
    def __init__(self):
        self.scratchpad: dict[str, Any] = dict()
        self.lock = asyncio.Lock()

    def _format(self) -> str:
        return json.dumps(self.scratchpad)

    async def add_todos(self, todos: list[str]):
        """Add todos to the scratchpad.

        Args:
            todos (list[str]): The todos to add.
        """
        async with self.lock:
            if "todos" not in self.scratchpad:
                self.scratchpad["todos"] = dict()
            for todo in todos:
                self.scratchpad["todos"][todo] = False

    async def mark_todos_as_done(self, todos: list[str]):
        """Mark todos as done.

        Args:
            todos (list[str]): The todos to mark as done.
        """
        async with self.lock:
            if "todos" not in self.scratchpad or not self.scratchpad["todos"]:
                raise ValueError("No todos to mark as done")
            for todo in todos:
                if todo in self.scratchpad["todos"] and todo:
                    self.scratchpad["todos"][todo] = True
                else:
                    # Optionally raise an error or log a warning if a todo doesn't exist
                    print(f"Warning: Todo '{todo}' not found in scratchpad.")

    async def add_notes(self, notes: list[dict | str]):
        """Add notes to the scratchpad.

        Args:
            notes (list[str]): The notes to add.
        """
        async with self.lock:
            if "notes" not in self.scratchpad:
                self.scratchpad["notes"] = list()
            for note in notes:
                if note:
                    self.scratchpad["notes"].append(note)

    async def get_notes(self) -> list[dict | str]:
        """Get the notes from the scratchpad.

        Returns:
            list[dict | str]: The notes.
        """
        async with self.lock:
            return self.scratchpad["notes"]

    async def get_todos(self) -> dict[str, bool]:
        """Get the todos from the scratchpad.

        Returns:
            dict[str, bool]: The todos.
        """
        async with self.lock:
            return self.scratchpad["todos"]
