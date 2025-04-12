from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def add(self, memory: str):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def clear(self):
        pass
