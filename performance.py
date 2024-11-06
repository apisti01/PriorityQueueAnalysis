import time
import random
import statistics
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from PriorityQueueItem import PriorityQueueItem
from PriorityQueueInterface import PriorityQueueInterface
from MaxHeapPriorityQueue import MaxHeapPriorityQueue
from LinkedListPriorityQueue import LinkedListPriorityQueue
from OrderedLinkedListPriorityQueue import OrderedLinkedListPriorityQueue


class Benchmark:
    def __init__(self, implementations, sizes=None):
        if sizes is None:
            sizes = [100, 1000, 2500, 5000, 7500, 10000]
        self.implementations = implementations
        self.sizes = sizes
        self.results = {}

    def run_single_test(self, impl_class, test_data: List[PriorityQueueItem],
                        test_type: str) -> float:
        """Run a single test and return execution time."""
        queue = impl_class()

        if test_type == "sequential_insert":
            start_time = time.perf_counter()
            for item in test_data:
                queue.insert(item)
            return time.perf_counter() - start_time

        elif test_type == "insert_extract_mix":
            # Test mixed operations
            start_time = time.perf_counter()
            ops_count = len(test_data)
            for i in range(ops_count):
                if random.random() < 0.7:  # 70% insert, 30% extract
                    queue.insert(test_data[i])
                elif not queue.is_empty():
                    queue.extract_max()
            return time.perf_counter() - start_time

        elif test_type == "peak_usage":
            # Test performance under peak load
            # First insert all items
            for item in test_data:
                queue.insert(item)

            # Then perform intensive mixed operations
            start_time = time.perf_counter()
            for _ in range(len(test_data) // 2):
                if random.random() < 0.5:
                    queue.insert(PriorityQueueItem(
                        priority=random.randint(1, 10000),
                        item=f"peak_test_{_}"
                    ))
                if not queue.is_empty():
                    queue.extract_max()
            return time.perf_counter() - start_time

        elif test_type == "extract_all":
            # First insert all items
            for item in test_data:
                queue.insert(item)

            # Test extraction performance
            start_time = time.perf_counter()
            while not queue.is_empty():
                queue.extract_max()
            return time.perf_counter() - start_time

    def generate_test_data(self, size: int, pattern: str = "random") -> List[PriorityQueueItem]:
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
        elif pattern == "clustered":
            # Generate data clustered around certain priority values
            clusters = [1000, 3000, 5000, 7000, 9000]
            return [PriorityQueueItem(
                priority=int(random.gauss(random.choice(clusters), 200)),
                item=f"value_{i}"
            ) for i in range(size)]

    def run_benchmarks(self, iterations: int = 5):
        """Run comprehensive benchmarks."""
        test_types = [
            "sequential_insert",
            "insert_extract_mix",
            "peak_usage",
            "extract_all"
        ]

        data_patterns = ["random", "ascending", "descending", "clustered"]

        results = {impl_name: {
            size: {
                test_type: {
                    pattern: [] for pattern in data_patterns
                } for test_type in test_types
            } for size in self.sizes
        } for _, impl_name in self.implementations}

        for size in self.sizes:
            print(f"\nTesting size: {size}")

            for pattern in data_patterns:
                print(f"  Data pattern: {pattern}")

                for iteration in range(iterations):
                    test_data = self.generate_test_data(size, pattern)

                    for impl_class, impl_name in self.implementations:
                        print(f"    Running {impl_name} (iteration {iteration + 1}/{iterations})")

                        for test_type in test_types:
                            try:
                                time_taken = self.run_single_test(
                                    impl_class, test_data.copy(), test_type
                                )
                                results[impl_name][size][test_type][pattern].append(time_taken)
                            except Exception as e:
                                print(f"Error in {impl_name} {test_type}: {e}")

        self.results = results
        return results

    def plot_results(self):
        """Create comprehensive visualization of benchmark results."""
        test_types = list(next(iter(next(iter(self.results.values())).values())).keys())
        data_patterns = list(next(iter(next(iter(next(iter(
            self.results.values())).values())).values())).keys())

        for test_type in test_types:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            fig.suptitle(f'Performance Analysis: {test_type}', fontsize=16)

            for idx, pattern in enumerate(data_patterns):
                ax = axes[idx // 2, idx % 2]

                for impl_name in self.results.keys():
                    # Extract data for this implementation
                    y_values = []
                    y_errors = []

                    for size in self.sizes:
                        times = self.results[impl_name][size][test_type][pattern]
                        y_values.append(statistics.mean(times))
                        y_errors.append(statistics.stdev(times) if len(times) > 1 else 0)

                    # Plot with error bars
                    ax.errorbar(self.sizes, y_values, yerr=y_errors,
                                label=impl_name, marker='o', capsize=5)

                ax.set_xlabel('Input Size')
                ax.set_ylabel('Time (seconds)')
                ax.set_title(f'{pattern.capitalize()} Data Pattern')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_xscale('log')
                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig(f'benchmark_{test_type}.png')
            plt.close()

    def generate_report(self):
        """Generate a detailed performance report."""
        report = []

        for impl_name in self.results.keys():
            for size in self.sizes:
                for test_type in self.results[impl_name][size].keys():
                    for pattern in self.results[impl_name][size][test_type].keys():
                        times = self.results[impl_name][size][test_type][pattern]
                        report.append({
                            'Implementation': impl_name,
                            'Size': size,
                            'Test Type': test_type,
                            'Data Pattern': pattern,
                            'Mean Time': statistics.mean(times),
                            'Std Dev': statistics.stdev(times) if len(times) > 1 else 0,
                            'Min Time': min(times),
                            'Max Time': max(times)
                        })

        return pd.DataFrame(report)


def main():
    implementations = [
        (MaxHeapPriorityQueue, "Max Heap"),
        (LinkedListPriorityQueue, "Unordered Linked List"),
        (OrderedLinkedListPriorityQueue, "Ordered Linked List")
    ]

    # Initialize and run benchmarks
    benchmark = Benchmark(implementations)
    benchmark.run_benchmarks()

    # Generate visualizations
    benchmark.plot_results()

    # Generate and save detailed report
    report = benchmark.generate_report()
    report.to_csv('priority_queue_benchmark_report.csv')

    # Print summary statistics
    print("\nSummary Statistics:")
    print(report.groupby(['Implementation', 'Test Type'])['Mean Time'].mean())


if __name__ == "__main__":
    main()
