from dataclasses import dataclass

import PriorityQueueItem
from Node import Node
from LinkedListPriorityQueue import LinkedListPriorityQueue


@dataclass
class OrderedLinkedListPriorityQueue(LinkedListPriorityQueue):

    def insert(self, item: PriorityQueueItem) -> None:
        """Inserts the item into the linked list in order of priority."""
        new_node = Node(item=item)
        if not self.head:
            self.head = new_node
        elif item.priority > self.head.priority():
            new_node.next = self.head
            self.head = new_node
        else:
            prev = None
            current = self.head
            while current and item.priority <= current.priority():
                prev = current
                current = current.next
            prev.next = new_node
            new_node.next = current

    def extract_max(self) -> PriorityQueueItem:
        """Removes and returns the item with the highest priority in the linked list."""
        if self.is_empty():
            raise IndexError("extract_max from an empty priority queue")

        item = self.head.item
        self.head = self.head.next
        return item

    def peek_max(self) -> PriorityQueueItem:
        """Returns the item with the highest priority without removing it."""
        if self.is_empty():
            raise IndexError("peek_max from an empty priority queue")
        return self.head.item
