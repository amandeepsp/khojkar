class Scratchpad:
    def __init__(self) -> None:
        self.scratchpad = {}

    def add(self, key: str, value: str) -> None:
        self.scratchpad[key] = value
