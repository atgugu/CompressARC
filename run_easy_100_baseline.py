"""
Script to run baseline evaluation on 100 easiest ARC tasks.
Modified from parallel_train.py to use only the selected easy tasks.
"""

import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch

import preprocessing
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import solve_task

# Getting all the task names, setting defaults and constants
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Function that can spawn processes that solve a puzzle, and schedule them on GPUs to take up each GPUs quota
def parallelize_runs(gpu_quotas, task_usages, n_iterations, verbose=False):
    """
    Runs a server that spawns processes to solve many ARC-AGI tasks in parallel.
    Args:
        gpu_quotas (list[float]): The max quota that each GPU has to use.
        task_usages (list[float]): The amount of quota that each task uses.
        n_iterations (int): The number of training iterations to use to solve each puzzle.
        verbose (bool): Whether or not to print tqdm bars to show progress.
    Returns:
        Dict[str, int]: The max memory allocated for every puzzle.
        Dict[str, list[Dict[str, list[list[int]]]]]: The guessed solution for every puzzle.
        float: The amount of time taken to solve all the puzzles in parallel.
    """
    t = time.time()
    gpu_quotas = gpu_quotas[:]
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]

    with multiprocessing.Manager() as manager:

        # Construct structures for inter-process communication
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        # Job monitoring loop
        while not all(tasks_finished):

            # Scan for errors
            if not error_queue.empty():
                raise ValueError(error_queue.get())

            # If a job finishes, release its quota
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])

            # If there is enough quota to start a new job, do it
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        args = (task_names[i], split, 1e20, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(task_names[i], 'started on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
            time.sleep(1)

        # Scan for errors
        if not error_queue.empty():
            raise ValueError(error_queue.get())

        # Save the solutions in the server process
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

    print(f"Running baseline evaluation on {n_tasks} easy tasks...")
    print(f"Task IDs: {task_names[:10]}... (showing first 10)")

    split = "training"

    # Measuring the amount of memory used for every task
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]
    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, 2, verbose=True)

    # Sort the tasks by decreasing memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)

    # Computing the solution for every task, while saturating memory
    n_steps = 2000
    safe_gpu_memory_quotas = [memory_quota - 4 * 1024**3 for memory_quota in gpu_memory_quotas]
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps, verbose=True)

    # Format the solutions and put into submission file
    with open('submission_easy_100.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)

    with open("submission_easy_100.json", "r") as f:
        contents = json.load(f)
    print(len(contents.keys()), 'puzzles solved.')
    print(n_steps, 'steps per puzzle.')
    print(time_taken, 'seconds.')
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
