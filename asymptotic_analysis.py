import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from scipy import stats

from PriorityQueueItem import PriorityQueueItem
from MaxHeapPriorityQueue import MaxHeapPriorityQueue
from LinkedListPriorityQueue import LinkedListPriorityQueue
from OrderedLinkedListPriorityQueue import OrderedLinkedListPriorityQueue


def generate_random_data(size: int) -> List[PriorityQueueItem]:
    """Generate random test data."""
    return [PriorityQueueItem(priority=random.randint(1, 10000), item=f"value_{i}")
            for i in range(size)]


def measure_performance(queue_class, data: List[PriorityQueueItem], batch_size) -> Dict[str, List[float]]:
    queue = queue_class()
    results = {
        'insert': [],
        'peek': [],
        'extract': []
    }

    # Measure insert and peek operations while building
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        # Measure insert
        start_time = time.perf_counter()
        for item in batch:
            queue.insert(item)
        end_time = time.perf_counter()
        operation_time = end_time - start_time
        results['insert'].append(operation_time)

        # Measure peek
        start_time = time.perf_counter()
        if not queue.is_empty():
            for _ in batch:
                queue.peek_max()
        end_time = time.perf_counter()
        operation_time = end_time - start_time
        results['peek'].append(operation_time)

    # Measure extract operations
    size = len(data)
    extracted = 0
    while not queue.is_empty():
        batch_size = min(batch_size, size - extracted)
        start_time = time.perf_counter()
        for _ in range(batch_size):
            if not queue.is_empty():
                queue.extract_max()
                extracted += 1
        end_time = time.perf_counter()
        operation_time = end_time - start_time
        results['extract'].append(operation_time)

    return results


def plot_results(results: Dict[str, Dict[str, List[float]]]):
    """
    Create detailed visualizations with:
    1. Separate figure for each implementation
    2. Subplots for insert, peek, extract operations
    3. trend analysis
    """
    plt.close('all')  # Close any existing plots

    for impl_name, impl_results in results.items():
        operations = ['insert', 'peek', 'extract']

        for idx, operation in enumerate(operations):
            # Create a new figure for each operation
            fig, ax_raw = plt.subplots(figsize=(15, 10))
            fig.suptitle(f'Detailed Performance Analysis: {impl_name} - {operation.capitalize()}\n', fontsize=16)

            x_values = range(1, len(impl_results[operation]) + 1)
            times = impl_results[operation]

            ax_raw.plot(x_values, times, label=f'{operation.capitalize()} Time', color='blue')

            # Trend line
            z = np.polyfit(x_values, times, 1)
            p = np.poly1d(z)
            ax_raw.plot(x_values, p(x_values), color='red', linestyle='--',
                        label=f'Trend Line (slope: {z[0]:.2e})')

            ax_raw.set_xlabel('Number of Operations')
            ax_raw.set_ylabel('Time (seconds)')
            ax_raw.legend()
            ax_raw.grid(True, alpha=0.3)

            plt.tight_layout()
            # Save figure with implementation name and operation
            safe_impl_name = impl_name.replace(' ', '_').lower()

            # Create a directory to store detailed plots
            if not os.path.exists('detailed_plots'):
                os.makedirs('detailed_plots')

            plt.savefig(f'detailed_plots/{safe_impl_name}_{operation}.png')
            plt.close(fig)


def remove_outliers(data: List[float], z_threshold: float = 4.0) -> List[float]:
    """
    Remove outliers using z-score method.
    Points with z-score greater than z_threshold are considered outliers.
    """
    z_scores = stats.zscore(data)
    return [x for x, z in zip(data, z_scores) if abs(z) < z_threshold]


def plot_smoothed_results(results: Dict[str, Dict[str, List[float]]]):
    """
    Create smoothed visualizations with:
    1. Separate figure for each implementation
    2. Subplots for insert, peek, extract operations
    3. Remove outliers and smooth the data
    4. Adjust the y-axis scaling to highlight the trends
    """
    plt.close('all')  # Close any existing plots

    for impl_name, impl_results in results.items():
        operations = ['insert', 'peek', 'extract']

        for idx, operation in enumerate(operations):
            # Create a new figure for each operation
            fig, ax_smooth = plt.subplots(figsize=(15, 10))
            fig.suptitle(f'Smoothed Performance Analysis: {impl_name} - {operation.capitalize()}\n', fontsize=16)

            # Remove outliers first
            times = impl_results[operation]
            cleaned_times = remove_outliers(times)

            # Generate x values accounting for removed points
            x_values = np.linspace(1, len(times), len(cleaned_times))

            # Exponential moving average to smooth the data
            smoothed_times = [cleaned_times[0]]
            alpha = 0.1  # Smoothing factor
            for t in cleaned_times[1:]:
                smoothed_times.append(alpha * t + (1 - alpha) * smoothed_times[-1])

            ax_smooth.plot(x_values, smoothed_times, label=f'{operation.capitalize()} Time', color='blue')

            # Trend line
            z = np.polyfit(x_values, smoothed_times, 1)
            p = np.poly1d(z)
            ax_smooth.plot(x_values, p(x_values), color='red', linestyle='--',
                           label=f'Trend Line (slope: {z[0]:.2e})')

            # Adjust y-axis scaling to highlight the trend, excluding the first 50 points
            max_value = max(smoothed_times[50:])
            min_value = min(smoothed_times[50:])
            if abs(z[0]) < 1e-6:
                max_value = max_value * 1.3
                min_value = min_value * 0.7
            ax_smooth.set_ylim(min_value * 0.9, max_value * 1.1)

            ax_smooth.set_xlabel('Number of Operations')
            ax_smooth.set_ylabel('Time (seconds)')
            ax_smooth.legend()
            ax_smooth.grid(True, alpha=0.3)

            plt.tight_layout()
            # Save figure with implementation name and operation
            safe_impl_name = impl_name.replace(' ', '_').lower()

            # Create a directory to store smoothed plots
            if not os.path.exists('smoothed_plots'):
                os.makedirs('smoothed_plots')

            plt.savefig(f'smoothed_plots/{safe_impl_name}_{operation}.png')
            plt.close(fig)


def analyse_asymptotic_behaviour():
    implementations = [
        (MaxHeapPriorityQueue, "Max Heap", 500000, 500),
        (LinkedListPriorityQueue, "Unordered Linked List", 30000, 30),
        (OrderedLinkedListPriorityQueue, "Ordered Linked List", 40000, 40)
    ]

    # Run tests for each implementation
    results = {}
    for impl_class, name, size, batch_size in implementations:
        print(f"Generating test data of size {size} for {name}...")
        data = generate_random_data(size)
        print(f"Testing {name}...")
        results[name] = measure_performance(impl_class, data.copy(), batch_size)

    # Create detailed visualizations
    for impl_name, impl_results in results.items():
        plot_results({impl_name: impl_results})

    plot_smoothed_results(results)


if __name__ == "__main__":
    analyse_asymptotic_behaviour()
