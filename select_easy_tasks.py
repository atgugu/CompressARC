"""
Script to identify the 100 easiest ARC tasks based on complexity heuristics.
"""

import json
import numpy as np

def calculate_task_complexity(task_data):
    """
    Calculate a complexity score for a task based on multiple factors.
    Lower score = easier task.
    """
    train_examples = task_data['train']
    test_examples = task_data['test']

    # Factor 1: Number of training examples (more examples might be easier)
    num_train = len(train_examples)

    # Factor 2: Grid sizes (smaller grids are easier)
    max_grid_size = 0
    total_grid_size = 0
    count = 0

    # Train examples have both input and output
    for example in train_examples:
        for grid_type in ['input', 'output']:
            grid = example[grid_type]
            height = len(grid)
            width = len(grid[0]) if height > 0 else 0
            size = height * width
            max_grid_size = max(max_grid_size, size)
            total_grid_size += size
            count += 1

    # Test examples only have input
    for example in test_examples:
        grid = example['input']
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        size = height * width
        max_grid_size = max(max_grid_size, size)
        total_grid_size += size
        count += 1

    avg_grid_size = total_grid_size / count if count > 0 else 0

    # Factor 3: Number of unique colors used
    unique_colors = set()
    for example in train_examples:
        for grid_type in ['input', 'output']:
            grid = example[grid_type]
            for row in grid:
                unique_colors.update(row)
    for example in test_examples:
        grid = example['input']
        for row in grid:
            unique_colors.update(row)
    num_colors = len(unique_colors)

    # Calculate complexity score (weighted combination)
    # Lower values indicate easier tasks
    complexity = (
        avg_grid_size * 1.0 +           # Average grid size
        max_grid_size * 0.5 +            # Max grid size
        num_colors * 5.0 +               # Number of colors
        (1.0 / (num_train + 1)) * 50.0  # Inverse of training examples
    )

    return complexity

def select_easy_tasks(challenges_file, n_tasks=100):
    """
    Select the n easiest tasks from the challenges file.
    """
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)

    # Calculate complexity for each task
    task_complexities = []
    for task_id, task_data in challenges.items():
        complexity = calculate_task_complexity(task_data)
        task_complexities.append((task_id, complexity))

    # Sort by complexity (ascending) and take the n easiest
    task_complexities.sort(key=lambda x: x[1])
    easy_task_ids = [task_id for task_id, _ in task_complexities[:n_tasks]]

    return easy_task_ids

if __name__ == '__main__':
    # Select 100 easiest tasks from training set
    easy_tasks = select_easy_tasks('dataset/arc-agi_training_challenges.json', n_tasks=100)

    # Save to a file
    with open('easy_tasks_100.json', 'w') as f:
        json.dump(easy_tasks, f, indent=2)

    print(f"Selected {len(easy_tasks)} easy tasks")
    print(f"First 10 task IDs: {easy_tasks[:10]}")
    print(f"Saved to easy_tasks_100.json")
