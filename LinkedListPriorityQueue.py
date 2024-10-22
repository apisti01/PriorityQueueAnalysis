# Concrete implementation of non-ordered linked list as a dataclass
from dataclasses import dataclass
from typing import Optional

from Node import Node
import PriorityQueueItem
import PriorityQueueInterface


@dataclass
class LinkedListPriorityQueue(PriorityQueueInterface):
    head: Optional[Node] = None

    def insert(self, item: PriorityQueueItem) -> None:
        """Inserts the item into the linked list without ordering."""
        new_node = Node(item=item)
        if not self.head:
            self.head = new_node
        else:
            # Insert new node at the beginning (for simplicity)
            new_node.next = self.head
            self.head = new_node

    def extract_max(self) -> PriorityQueueItem:
        """Removes and returns the item with the highest priority in the linked list."""
        if self.is_empty():
            raise IndexError("extract_max from an empty priority queue")

        # Traverse the linked list to find the node with the max priority
        max_node = self.head
        max_node_prev = None
        prev = None
        current = self.head

        while current:
            if current.priority > max_node.priority:
                max_node = current
                max_node_prev = prev
            prev = current
            current = current.next

        # Remove the max node from the linked list
        if max_node_prev is None:  # The max node is the head
            self.head = self.head.next
        else:
            max_node_prev.next = max_node.next

        return max_node.item

    def peek_max(self) -> PriorityQueueItem:
        """Returns the item with the highest priority without removing it."""
        if self.is_empty():
            raise IndexError("peek_max from an empty priority queue")

        # Traverse the linked list to find the node with the max priority
        max_node = self.head
        current = self.head

        while current:
            if current.priority > max_node.priority:
                max_node = current
            current = current.next

        return max_node.item

    def is_empty(self) -> bool:
        """Checks if the linked list is empty."""
        return self.head is None
