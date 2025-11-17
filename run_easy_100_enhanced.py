"""
Enhanced baseline evaluation on 100 easiest ARC tasks.
Uses constraint-based solution selection for improved accuracy.

This script is identical to run_easy_100_baseline.py but uses the EnhancedLogger
from enhanced_solution_selection.py instead of the basic Logger.
"""

import os
import sys
import time
import json
import multiprocessing

import numpy as np
import torch

import preprocessing
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import enhanced_solution_selection as solution_selection  # Use enhanced version
import visualization
import solve_task

# Getting all the task names, setting defaults and constants
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def solve_task_enhanced(task_name, split, max_memory, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue):
    """
    Enhanced version of solve_task that uses constraint-based solution selection.

    Args:
        task_name (str): Name of the task to solve.
        split (str): Dataset split ('training', 'evaluation', or 'test').
        max_memory (float): Maximum memory allowed for this task.
        n_iterations (int): Number of training iterations.
        gpu_id (int): GPU to use.
        memory_dict (dict): Shared dictionary for memory usage.
        solutions_dict (dict): Shared dictionary for solutions.
        error_queue (Queue): Queue for error messages.
    """
    try:
        torch.cuda.set_device(gpu_id)

        # Preprocess task
        task = preprocessing.preprocess_tasks(split, task_names=[task_name])[0]

        # Create model
        model = arc_compressor.ARCCompressor(task)

        # Create optimizer
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))

        # Create ENHANCED logger with constraint validation
        logger = solution_selection.EnhancedLogger(task, use_constraints=True, n_candidates=20)

        # Train
        for train_step in range(n_iterations):
            train.take_step(task, model, optimizer, train_step, logger)

        # Get solutions
        solution_dict = {
            'attempt_1': list(logger.solution_most_frequent) if logger.solution_most_frequent else [],
            'attempt_2': list(logger.solution_second_most_frequent) if logger.solution_second_most_frequent else []
        }

        # Format for submission
        formatted_solution = []
        for test_idx in range(task.n_test):
            attempts = {}
            if logger.solution_most_frequent and test_idx < len(logger.solution_most_frequent):
                attempts['attempt_1'] = [list(row) for row in logger.solution_most_frequent[test_idx]]
            if logger.solution_second_most_frequent and test_idx < len(logger.solution_second_most_frequent):
                attempts['attempt_2'] = [list(row) for row in logger.solution_second_most_frequent[test_idx]]
            formatted_solution.append(attempts)

        solutions_dict[task_name] = formatted_solution

        # Record memory
        memory_used = torch.cuda.max_memory_allocated(gpu_id)
        memory_dict[task_name] = memory_used

        torch.cuda.reset_peak_memory_stats(gpu_id)

    except Exception as e:
        error_queue.put(f"Error in task {task_name}: {str(e)}")


def parallelize_runs(gpu_quotas, task_usages, n_iterations, task_names, split, n_gpus, n_cpus, verbose=False):
    """
    Runs a server that spawns processes to solve many ARC-AGI tasks in parallel.
    """
    n_tasks = len(task_names)
    t = time.time()
    gpu_quotas = gpu_quotas[:]
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]

    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())

            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])

            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        args = (task_names[i], split, 1e20, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task_enhanced, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(task_names[i], 'started on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
            time.sleep(1)

        if not error_queue.empty():
            raise ValueError(error_queue.get())

        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)

    time_taken = time.time() - t
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')
    return memory_dict, solutions_dict, time_taken


if __name__ == '__main__':
    start_time = time.time()

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Load the selected 100 easy tasks
    with open('easy_tasks_100.json', 'r') as f:
        task_names = json.load(f)
    n_tasks = len(task_names)

    print("=" * 70)
    print("ENHANCED BASELINE EVALUATION ON 100 EASY ARC TASKS")
    print("=" * 70)
    print(f"Using constraint-based solution selection with diverse sampling")
    print(f"Number of tasks: {n_tasks}")
    print(f"Number of GPUs: {n_gpus}")
    print(f"Number of CPUs: {n_cpus}")
    print(f"Task IDs: {task_names[:10]}... (showing first 10)")
    print("=" * 70)

    split = "training"

    # Measuring the amount of memory used for every task
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]

    print("\nPhase 1: Measuring memory usage...")
    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, 2, task_names, split, n_gpus, n_cpus, verbose=True)

    # Sort the tasks by decreasing memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names_sorted, task_memory_usages = zip(*tasks)

    # Computing the solution for every task, while saturating memory
    n_steps = 2000
    safe_gpu_memory_quotas = [memory_quota - 4 * 1024**3 for memory_quota in gpu_memory_quotas]

    print(f"\nPhase 2: Running enhanced evaluation ({n_steps} steps per task)...")
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps,
                                                      list(task_names_sorted), split, n_gpus, n_cpus, verbose=True)

    # Format the solutions and put into submission file
    with open('submission_easy_100_enhanced.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Evaluation complete!")
    print(f"Tasks solved: {len(solutions_dict)}")
    print(f"Steps per task: {n_steps}")
    print(f"Evaluation time: {time_taken:.2f} seconds")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: submission_easy_100_enhanced.json")
    print("=" * 70)
