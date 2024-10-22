from dataclasses import dataclass, field
from typing import Optional

from PriorityQueueItem import PriorityQueueItem


@dataclass
class Node:
    item: PriorityQueueItem
    next: Optional["Node"] = field(default=None, init=False)

    def priority(self) -> int:
        return self.item.priority
