from dataclasses import dataclass, field
from typing import Any


# Dataclass to represent the queue item with a priority
@dataclass(order=True)
class PriorityQueueItem:
    priority: int
    item: Any = field(compare=False)