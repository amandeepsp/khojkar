import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

console = Console()


class Scratchpad:
    def __init__(self):
        self.scratchpad: dict[str, Any] = dict()

        self.lock = asyncio.Lock()

    def _format(self) -> str:
        formatted_scratchpad = ""
        if "todos" in self.scratchpad:
            formatted_scratchpad += "\n## TODOS\n"
            for todo in self.scratchpad["todos"]:
                formatted_scratchpad += (
                    f"- [{'x' if self.scratchpad['todos'][todo] else ' '}] {todo}\n"
                )
        if "notes" in self.scratchpad:
            formatted_scratchpad += "\n## NOTES\n"
            formatted_scratchpad += self.scratchpad["notes"]

        console.print(Markdown(formatted_scratchpad))
        return formatted_scratchpad

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
            return self._format()

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
            return self._format()

    async def add_note(self, note: str):
        """Add a note to the scratchpad.

        Args:
            note (str): The note to add.
        """
        async with self.lock:
            if "notes" not in self.scratchpad:
                self.scratchpad["notes"] = ""
            self.scratchpad["notes"] += f"\n{note}"
            return f"""
                # Notes
                {self.scratchpad["notes"]}
            """

    async def get_notes(self) -> str:
        """Get the notes from the scratchpad.

        Returns:
            str: The notes.
        """
        return self.scratchpad["notes"]
