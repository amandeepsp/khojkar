import logging

logger = logging.getLogger(__name__)


class MessagesMemory:
    """
    A memory that is used to store the context of the conversation.
    Truncates in a FIFO manner.
    """

    def __init__(self, max_tokens: int):
        self.system_prompt = None
        self.max_tokens = max_tokens
        self.messages = []
        self.total_tokens = 0

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.messages = [{"role": "user", "content": system_prompt}]
        self.total_tokens = self._num_tokens_from_string(system_prompt)

    def add_formatted_system_prompt(self, system_prompt, **kwargs):
        _flattened_kwargs = {k: str(v) for k, v in kwargs.items() if v is not None}
        _formatted_system_prompt = system_prompt.format(**_flattened_kwargs)
        self.system_prompt = _formatted_system_prompt
        self.messages = [{"role": "user", "content": _formatted_system_prompt}]
        self.total_tokens = self._num_tokens_from_string(_formatted_system_prompt)

    def _num_tokens_from_string(self, string: str | None) -> int:
        """Estimate the number of tokens in a string."""
        if string is None:
            return 0
        return len(string) // 4

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

    def get_all(self):
        return self.messages

    def clear(self):
        self.messages = []
        self.total_tokens = 0

    def __len__(self):
        return len(self.messages)
