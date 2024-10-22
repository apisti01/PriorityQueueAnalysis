from PriorityQueueItem import PriorityQueueItem
from abc import ABC, abstractmethod


# Abstract base class to define the interface
class PriorityQueueInterface(ABC):

    @abstractmethod
    def insert(self, item: PriorityQueueItem) -> None:
        pass

    @abstractmethod
    def extract_max(self) -> PriorityQueueItem:
        pass

    @abstractmethod
    def peek_max(self) -> PriorityQueueItem:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass