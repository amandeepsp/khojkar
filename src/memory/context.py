import logging
from typing import override

from memory.memory import Memory

logger = logging.getLogger(__name__)


class InContextMemory(Memory):
    """
    A memory that is used to store the context of the conversation.
    Truncates in a FIFO manner.
    """

    def __init__(self, system_prompt: str, max_tokens: int):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages = [{"role": "user", "content": system_prompt}]
        self.total_tokens = self._num_tokens_from_string(system_prompt)

    def _num_tokens_from_string(self, string: str | None) -> int:
        """Estimate the number of tokens in a string."""
        if string is None:
            return 0
        return len(string) // 4

    @override
    def add(self, message):
        content = message.get("content", "")
        self.messages.append(message)
        self.total_tokens += self._num_tokens_from_string(content)
        self._prune()

    def _prune(self):
        while self.total_tokens > self.max_tokens and len(self.messages) > 1:
            removed_message = self.messages.pop(1)  # Keep system prompt at index 0
            removed_content = removed_message.get("content", "")
            self.total_tokens -= self._num_tokens_from_string(removed_content)
            logger.info(f"Context pruned to {len(self.messages)} messages")

    @override
    def get_all(self):
        return self.messages

    @override
    def clear(self):
        self.messages = [{"role": "user", "content": self.system_prompt}]
        self.total_tokens = self._num_tokens_from_string(self.system_prompt)

    @override
    def query(self, query: str):
        raise NotImplementedError("InContextMemory does not support querying")

    def __len__(self):
        return len(self.messages)
