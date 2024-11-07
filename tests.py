import os
import time
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

from PriorityQueueItem import PriorityQueueItem
from PriorityQueueInterface import PriorityQueueInterface
from Node import Node
from MaxHeapPriorityQueue import MaxHeapPriorityQueue
from LinkedListPriorityQueue import LinkedListPriorityQueue
from OrderedLinkedListPriorityQueue import OrderedLinkedListPriorityQueue


def generate_test_data(size: int, pattern: str = "random") -> List[PriorityQueueItem]:
    """Generate test data with different patterns."""
    if pattern == "random":
        return [PriorityQueueItem(priority=random.randint(1, 10000),
                                  item=f"value_{i}") for i in range(size)]
    elif pattern == "ascending":
        return [PriorityQueueItem(priority=i,
                                  item=f"value_{i}") for i in range(size)]
    elif pattern == "descending":
        return [PriorityQueueItem(priority=size - i,
                                  item=f"value_{i}") for i in range(size)]


def test_implementation(queue_class, data: List[PriorityQueueItem]) -> tuple[float, float, float, float]:
    """Test a specific implementation and return timing results."""
    queue = queue_class()

    # Measure insert time
    start_time = time.perf_counter()
    for item in data:
        queue.insert(item)
    insert_time = time.perf_counter() - start_time

    # Measure peek time
    start_time = time.perf_counter()
    for _ in range(len(data) // 10):  # Test peek 10% of size times
        queue.peek_max()
    peek_time = (time.perf_counter() - start_time) * 10  # Normalize to full size

    # Measure extract time
    start_time = time.perf_counter()
    while not queue.is_empty():
        queue.extract_max()
    extract_time = time.perf_counter() - start_time

    # Measure mixed use time
    ops_count = len(data)
    start_time = time.perf_counter()
    for i in range(ops_count):
        if random.random() < 0.6:  # 60% insert, 40% extract
            queue.insert(data[i])
        elif not queue.is_empty():
            queue.extract_max()
    mixed_time = time.perf_counter() - start_time

    return insert_time, peek_time, extract_time, mixed_time


def run_performance_tests(sizes):
    implementations = [
        (MaxHeapPriorityQueue, "Max Heap"),
        (LinkedListPriorityQueue, "Unordered Linked List"),
        (OrderedLinkedListPriorityQueue, "Ordered Linked List")
    ]
    patterns = ['random', 'ascending', 'descending']
    operations = ['insert', 'peek', 'extract', 'mixed']

    results = {name: {pattern: {operation: [] for operation in operations} for pattern in patterns} for _, name in
               implementations}

    for size in sizes:
        print(f"\nTesting with size {size}")

        for pattern in ["random", "ascending", "descending"]:
            print(f"Generating {pattern} data...")
            data = generate_test_data(size, pattern)

            for impl_class, name in implementations:
                print(f"Testing {name} with {pattern} data...")
                insert_time, peek_time, extract_time, mixed_time = test_implementation(impl_class, data.copy())
                results[name][pattern]['insert'].append(insert_time)
                results[name][pattern]['peek'].append(peek_time)
                results[name][pattern]['extract'].append(extract_time)
                results[name][pattern]['mixed'].append(mixed_time)

    return results


def plot_results(sizes: List[int], results: Dict[str, Dict[str, Dict[str, List[float]]]]):
    """Create visualizations of the performance results."""
    operations = ['insert', 'peek', 'extract', 'mixed']
    patterns = ['random', 'ascending', 'descending']

    for pattern in patterns:
        for idx, operation in enumerate(operations):
            fig_normal, ax = plt.subplots(figsize=(15, 10))
            fig_normal.suptitle(
                f'Priority Queue Implementation Performance Comparison (Normal Scale) - {pattern.capitalize()} Data - {operation.capitalize()} Operation')

            for impl_name in results.keys():
                ax.plot(sizes, results[impl_name][pattern][operation], marker='o', label=impl_name)

            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True)
            ax.legend()

            plt.tight_layout()
            if not os.path.exists('performance_normal'):
                os.makedirs('performance_normal')
            plt.savefig(f'performance_normal/priority_queue_performance_normal_{pattern}_{operation}.png')
            plt.close(fig_normal)

        # Log scale plot
        for idx, operation in enumerate(operations):
            fig_log, ax = plt.subplots(figsize=(15, 10))
            fig_log.suptitle(
                f'Priority Queue Implementation Performance Comparison (Log Scale) - {pattern.capitalize()} Data - {operation.capitalize()} Operation')

            for impl_name in results.keys():
                ax.plot(sizes, results[impl_name][pattern][operation], marker='o', label=impl_name)

            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True)
            ax.legend()
            ax.set_yscale('log')

            plt.tight_layout()
            if not os.path.exists('performance_log'):
                os.makedirs('performance_log')
            plt.savefig(f'performance_log/priority_queue_performance_log_{pattern}_{operation}.png')
            plt.close(fig_log)


def verify_correctness(size: int = 1000):
    """Verify that all implementations maintain correct ordering."""
    print("\nVerifying correctness of implementations...")
    data = generate_test_data(size)
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
    sizes = [50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    results = run_performance_tests(sizes)

    # Create and save visualization
    plot_results(sizes, results)


if __name__ == "__main__":
    main()
