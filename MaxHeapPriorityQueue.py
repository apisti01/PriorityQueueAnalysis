from dataclasses import dataclass, field
from typing import List

from PriorityQueueItem import PriorityQueueItem
from PriorityQueueInterface import PriorityQueueInterface


@dataclass
class MaxHeapPriorityQueue(PriorityQueueInterface):
    """
    MaxHeapPriorityQueue class implements the PriorityQueueInterface using a max heap.
    """
    _heap: List[PriorityQueueItem] = field(default_factory=list, init=False)

    def _parent(self, index: int) -> int:
        return (index - 1) // 2

    def _left(self, index: int) -> int:
        return 2 * index + 1

    def _right(self, index: int) -> int:
        return 2 * index + 2

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def _heapify_up(self, index: int) -> None:
        while index > 0:
            parent_index = self._parent(index)
            if self._heap[index].priority > self._heap[parent_index].priority:
                self._swap(index, parent_index)
                index = parent_index
            else:
                break

    def _heapify_down(self, index: int) -> None:
        while True:
            largest = index
            left_index = self._left(index)
            right_index = self._right(index)

            if left_index < len(self._heap) and self._heap[left_index].priority > self._heap[largest].priority:
                largest = left_index

            if right_index < len(self._heap) and self._heap[right_index].priority > self._heap[largest].priority:
                largest = right_index

            if largest != index:
                self._swap(index, largest)
                index = largest
            else:
                break

    def insert(self, item: PriorityQueueItem) -> None:
        self._heap.append(item)
        self._heapify_up(len(self._heap) - 1)

    def extract_max(self) -> PriorityQueueItem:
        if self.is_empty():
            raise IndexError("extract_max from an empty priority queue")
        max_item = self._heap[0]
        # Move the last item to the root and heapify down
        self._heap[0] = self._heap.pop()  # Pop the last item and place it at the root
        if not self.is_empty():
            self._heapify_down(0)
        return max_item

    def peek_max(self) -> PriorityQueueItem:
        if self.is_empty():
            raise IndexError("peek_max from an empty priority queue")
        return self._heap[0]

    def is_empty(self) -> bool:
        return len(self._heap) == 0
