from PriorityQueueItem import PriorityQueueItem
from MaxHeapPriorityQueue import MaxHeapPriorityQueue
from LinkedListPriorityQueue import LinkedListPriorityQueue
from OrderedLinkedListPriorityQueue import OrderedLinkedListPriorityQueue


def test_priority_queue(pq_class):
    pq = pq_class()
    pq.insert(PriorityQueueItem(10, "task1"))
    pq.insert(PriorityQueueItem(15, "task2"))
    pq.insert(PriorityQueueItem(5, "task3"))
    pq.insert(PriorityQueueItem(20, "task4"))

    print("Max item:", pq.peek_max())  # Should print "task4"
    print("Extracted max item:", pq.extract_max())  # Should remove and return "task4"
    print("Next max item:", pq.peek_max())  # Should print "task2"
    print("\n")

    pq.extract_max()
    pq.extract_max()
    pq.extract_max()
    print(pq)


def main():
    pq_classes = [
        MaxHeapPriorityQueue,
        LinkedListPriorityQueue,
        OrderedLinkedListPriorityQueue
    ]

    for pq_class in pq_classes:
        print(f"Testing {pq_class.__name__}:")
        test_priority_queue(pq_class)


if __name__ == "__main__":
    main()
