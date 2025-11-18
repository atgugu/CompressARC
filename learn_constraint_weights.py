"""
Learn Constraint Weights from Validation Set

This script performs meta-learning to optimize constraint weights:
1. Split tasks into train (240) and validation (80) sets
2. Train models on validation tasks
3. Collect constraint features from candidates
4. Learn optimal weights to maximize selection accuracy
5. Save learned weights for use in evaluation

Usage:
    python learn_constraint_weights.py --method gradient --n_val_tasks 80

Methods:
    - logistic: Fast, requires sklearn
    - grid_search: Exhaustive, slow
    - gradient: Flexible, uses PyTorch
"""

import argparse
import json
import time
import numpy as np
import torch
import multiprocessing

import preprocessing
import arc_compressor
import enhanced_solution_selection as solution_selection
import train
import meta_learning


def train_single_task(task_name, split, n_iterations, gpu_id=0):
    """
    Train a model on a single task and return the logger with candidates.

    Args:
        task_name: Name of the task
        split: Dataset split
        n_iterations: Number of training iterations
        gpu_id: GPU to use

    Returns:
        (task, logger) tuple
    """
    torch.cuda.set_device(gpu_id)

    # Preprocess task
    task = preprocessing.preprocess_tasks(split, task_names=[task_name])[0]

    # Create model
    model = arc_compressor.ARCCompressor(task)

    # Create optimizer
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))

    # Create enhanced logger (NO learned weights yet - we're collecting data!)
    logger = solution_selection.EnhancedLogger(task, use_constraints=True, n_candidates=20)

    # Train
    for train_step in range(n_iterations):
        train.take_step(task, model, optimizer, train_step, logger)

    return task, logger


def main():
    parser = argparse.ArgumentParser(description='Learn constraint weights from validation set')
    parser.add_argument('--method', type=str, default='gradient',
                       choices=['logistic', 'grid_search', 'gradient'],
                       help='Weight learning method')
    parser.add_argument('--n_val_tasks', type=int, default=80,
                       help='Number of validation tasks')
    parser.add_argument('--n_iterations', type=int, default=1500,
                       help='Training iterations per task')
    parser.add_argument('--split', type=str, default='training',
                       help='Dataset split to use')
    parser.add_argument('--output', type=str, default='learned_weights.json',
                       help='Output file for learned weights')
    parser.add_argument('--save_dataset', type=str, default='meta_dataset.pkl',
                       help='Save collected meta-learning dataset')

    args = parser.parse_args()

    print("=" * 70)
    print("CONSTRAINT WEIGHT META-LEARNING")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Validation tasks: {args.n_val_tasks}")
    print(f"Training iterations: {args.n_iterations}")
    print("=" * 70)

    # Load all task names
    with open(f'dataset/arc-agi_{args.split}_challenges.json', 'r') as f:
        all_tasks = list(json.load(f).keys())

    # Split into train and validation
    np.random.seed(42)
    np.random.shuffle(all_tasks)

    val_task_names = all_tasks[:args.n_val_tasks]
    train_task_names = all_tasks[args.n_val_tasks:]

    print(f"\nTask split:")
    print(f"  Training: {len(train_task_names)} tasks")
    print(f"  Validation: {len(val_task_names)} tasks")

    # Save task splits
    with open('task_splits.json', 'w') as f:
        json.dump({
            'train': train_task_names,
            'validation': val_task_names
        }, f, indent=2)
    print(f"  Saved to: task_splits.json")

    # Train models on validation tasks and collect data
    print(f"\nPhase 1: Training models on validation tasks...")
    print(f"  This will take approximately {args.n_val_tasks * args.n_iterations * 0.05 / 60:.1f} minutes")

    tasks = []
    loggers = []

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("\nERROR: No CUDA GPUs available. This script requires GPU.")
        print("Please run on a machine with CUDA-enabled GPU.")
        return

    print(f"  Using {n_gpus} GPU(s)")

    for i, task_name in enumerate(val_task_names):
        gpu_id = i % n_gpus  # Round-robin GPU assignment

        print(f"  [{i+1}/{len(val_task_names)}] Training {task_name} on GPU {gpu_id}...")

        task, logger = train_single_task(task_name, args.split, args.n_iterations, gpu_id)

        tasks.append(task)
        loggers.append(logger)

    print(f"  Completed training on {len(tasks)} tasks")

    # Collect meta-learning dataset
    print(f"\nPhase 2: Collecting meta-learning dataset...")

    solutions_file = f'dataset/arc-agi_{args.split}_solutions.json'
    dataset = meta_learning.collect_meta_learning_data(tasks, loggers, solutions_file)

    # Save dataset
    dataset.save(args.save_dataset)
    print(f"  Dataset saved to: {args.save_dataset}")

    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\n  Dataset statistics:")
    print(f"    Total samples: {stats['n_samples']}")
    print(f"    Positive (correct): {stats['n_positive']}")
    print(f"    Negative (incorrect): {stats['n_negative']}")
    print(f"    Positive rate: {stats['positive_rate']:.3f}")

    # Learn weights
    print(f"\nPhase 3: Learning constraint weights using '{args.method}' method...")

    learner = meta_learning.ConstraintWeightLearner(method=args.method)

    if args.method == 'logistic':
        weights = learner.fit(dataset, C=1.0, max_iter=1000)
    elif args.method == 'grid_search':
        weights = learner.fit(dataset, n_top_weights=3)
    elif args.method == 'gradient':
        weights = learner.fit(dataset, lr=0.01, n_epochs=500, l2_reg=0.01)

    # Save weights
    learner.save_weights(args.output)

    print(f"\n" + "=" * 70)
    print("META-LEARNING COMPLETE")
    print("=" * 70)
    print(f"Learned weights saved to: {args.output}")
    print(f"Meta-learning dataset saved to: {args.save_dataset}")
    print(f"Task splits saved to: task_splits.json")
    print("\nNext steps:")
    print(f"  1. Run evaluation with learned weights:")
    print(f"     python run_easy_100_meta.py --weights {args.output}")
    print(f"  2. Compare to baseline:")
    print(f"     python score_easy_100_baseline.py")
    print("=" * 70)


if __name__ == '__main__':
    start_time = time.time()

    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_dtype(torch.float32)

    main()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
