import time
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from PriorityQueueItem import PriorityQueueItem
from PriorityQueueInterface import PriorityQueueInterface
from Node import Node
from MaxHeapPriorityQueue import MaxHeapPriorityQueue
from LinkedListPriorityQueue import LinkedListPriorityQueue
from OrderedLinkedListPriorityQueue import OrderedLinkedListPriorityQueue


def generate_random_data(size: int) -> List[PriorityQueueItem]:
    """Generate random test data."""
    return [PriorityQueueItem(priority=random.randint(1, 10000), item=f"value_{i}") for i in range(size)]


def test_implementation(queue_class, data: List[PriorityQueueItem]) -> Tuple[float, float, float]:
    """Test a specific implementation and return timing results."""
    queue = queue_class()

    # Measure insert time
    start_time = time.time()
    for item in data:
        queue.insert(item)
    insert_time = time.time() - start_time

    # Measure peek time
    start_time = time.time()
    for _ in range(len(data) // 10):  # Test peek 10% of size times
        queue.peek_max()
    peek_time = (time.time() - start_time) * 10  # Normalize to full size

    # Measure extract time
    start_time = time.time()
    while not queue.is_empty():
        queue.extract_max()
    extract_time = time.time() - start_time

    return insert_time, peek_time, extract_time


def run_performance_tests():
    """Run comprehensive performance tests on all implementations."""
    implementations = [
        (MaxHeapPriorityQueue, "Max Heap"),
        (LinkedListPriorityQueue, "Unordered Linked List"),
        (OrderedLinkedListPriorityQueue, "Ordered Linked List")
    ]

    sizes = [100, 500, 1000, 5000, 10000]
    results = {name: {
        'insert': [], 'peek': [], 'extract': [], 'total': []
    } for _, name in implementations}

    for size in sizes:
        print(f"\nTesting with size {size}")
        data = generate_random_data(size)

        for impl_class, name in implementations:
            print(f"Testing {name}...")
            insert_time, peek_time, extract_time = test_implementation(impl_class, data.copy())
            total_time = insert_time + peek_time + extract_time

            results[name]['insert'].append(insert_time)
            results[name]['peek'].append(peek_time)
            results[name]['extract'].append(extract_time)
            results[name]['total'].append(total_time)

    return sizes, results


def plot_results(sizes: List[int], results: dict):
    """Create visualizations of the performance results."""
    operations = ['insert', 'peek', 'extract', 'total']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Priority Queue Implementation Performance Comparison')

    for idx, operation in enumerate(operations):
        ax = axes[idx // 2, idx % 2]
        for impl_name in results.keys():
            ax.plot(sizes, results[impl_name][operation],
                    marker='o', label=impl_name)

        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{operation.capitalize()} Operation Performance')
        ax.grid(True)
        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    return fig


def verify_correctness(size: int = 1000):
    """Verify that all implementations maintain correct ordering."""
    print("\nVerifying correctness of implementations...")
    data = generate_random_data(size)
    implementations = [
        (MaxHeapPriorityQueue(), "Max Heap"),
        (LinkedListPriorityQueue(), "Unordered Linked List"),
        (OrderedLinkedListPriorityQueue(), "Ordered Linked List")
    ]

    for queue, name in implementations:
        # Insert all items
        for item in data:
            queue.insert(item)

        # Extract all items and verify they come out in descending order
        previous = float('inf')
        is_correct = True
        count = 0

        while not queue.is_empty():
            current = queue.extract_max()
            if current.priority > previous:
                is_correct = False
                break
            previous = current.priority
            count += 1

        print(f"{name}: {'✓ Correct' if is_correct and count == len(data) else '✗ Incorrect'}")


def main():
    # First verify correctness
    verify_correctness()

    # Then run performance tests
    print("\nRunning performance tests...")
    sizes, results = run_performance_tests()

    # Create and save visualization
    fig = plot_results(sizes, results)
    plt.savefig('priority_queue_performance.png')
    plt.close()

    # Print summary of findings
    print("\nPerformance Summary:")
    print("-" * 50)
    for impl_name in results.keys():
        print(f"\n{impl_name}:")
        print(
            f"Average insert time: {sum(results[impl_name]['insert']) / len(results[impl_name]['insert']):.6f} seconds")
        print(f"Average peek time: {sum(results[impl_name]['peek']) / len(results[impl_name]['peek']):.6f} seconds")
        print(
            f"Average extract time: {sum(results[impl_name]['extract']) / len(results[impl_name]['extract']):.6f} seconds")


if __name__ == "__main__":
    main()