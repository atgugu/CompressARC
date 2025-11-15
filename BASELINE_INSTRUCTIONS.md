# Baseline Evaluation on 100 Easy ARC Tasks

This document describes how to run a baseline evaluation on 100 of the easiest ARC tasks.

## What Has Been Prepared

1. **Task Selection** (`easy_tasks_100.json`):
   - 100 easiest ARC tasks have been selected from the training set
   - Selection based on complexity heuristics:
     - Average and maximum grid size
     - Number of unique colors
     - Number of training examples
   - Lower complexity = smaller grids, fewer colors, more training examples

2. **Evaluation Script** (`run_easy_100_baseline.py`):
   - Modified version of `parallel_train.py`
   - Runs only on the selected 100 easy tasks
   - Uses 2000 training steps per task
   - Parallelizes across available GPUs

3. **Scoring Script** (`score_easy_100_baseline.py`):
   - Scores the submission against ground truth
   - Provides detailed metrics including accuracy

## Requirements

- CUDA-enabled GPU (NVIDIA)
- Python 3.11+
- All dependencies from `requirements.txt` (already installed)

## How to Run the Baseline

### Step 1: Check GPU Availability

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Run the Baseline Evaluation

```bash
python run_easy_100_baseline.py
```

This will:
- Load the 100 easy tasks
- Measure memory usage for each task
- Train a model for each task (2000 steps)
- Save results to `submission_easy_100.json`

**Expected time**: Depends on GPU, but approximately 15-30 minutes per task on an RTX 4070.
For 100 tasks running in parallel, total time will vary based on GPU count and memory.

### Step 3: Score the Results

```bash
python score_easy_100_baseline.py
```

This will:
- Compare predictions against ground truth
- Calculate accuracy and average score
- Save detailed results to `baseline_results_easy_100.json`

## Output Files

- `submission_easy_100.json`: Model predictions for all 100 tasks
- `baseline_results_easy_100.json`: Detailed scoring results including:
  - Total score
  - Average score
  - Number of tasks fully correct
  - Accuracy percentage
  - Individual task scores

## Expected Baseline Performance

The baseline performance will depend on the model architecture and training setup.
This evaluation will establish a starting point for improvements.

## Selected Easy Tasks (First 10)

```
794b24be, 44f52bb0, 25ff71a9, 8be77c9e, c9e6f938,
6e02f1e3, 6d0aefbc, a85d4709, ed36ccf7, aedd82e4
```

## Notes

- The script uses multiprocessing to parallelize across GPUs
- Memory usage is measured first to optimize task scheduling
- Tasks are sorted by memory usage to maximize GPU utilization
- The evaluation runs the same code as the main `parallel_train.py` but only on 100 selected tasks
