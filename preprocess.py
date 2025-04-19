#!/usr/bin/env python
"""
CoDocBench Data Preprocessing Script

This script processes the CoDocBench dataset to create optimized train/val/test splits
with improved data filtering and balancing for documentation improvement models.
"""

import json
import os
import random
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import difflib
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Set


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CoDocBench dataset for better training")
    parser.add_argument("--input", type=str, default="codocbench/dataset/codocbench.jsonl",
                      help="Path to the raw JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="processed_dataset",
                      help="Directory to save processed dataset files")
    parser.add_argument("--train_size", type=float, default=0.7,
                      help="Proportion of data for training (default: 0.7)")
    parser.add_argument("--val_size", type=float, default=0.15,
                      help="Proportion of data for validation (default: 0.15)")
    parser.add_argument("--test_size", type=float, default=0.15,
                      help="Proportion of data for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--min_code_change", type=int, default=1,
                      help="Minimum number of lines that must change in code (default: 1)")
    parser.add_argument("--min_doc_change", type=int, default=1,
                      help="Minimum number of lines that must change in docstring (default: 1)")
    parser.add_argument("--max_length", type=int, default=1024,
                      help="Maximum token length to keep (default: 1024)")
    parser.add_argument("--balance_projects", action="store_true",
                      help="Balance examples across different projects")
    parser.add_argument("--project_aware_split", action="store_true",
                      help="Keep examples from the same project in the same split")
    return parser.parse_args()


def load_jsonl(file_path):
    """Load JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error parsing JSON line: {line}")
    return data


def compute_change_stats(example):
    """Extract useful stats about the code and docstring changes."""
    # Get before/after versions
    if len(example['version_data']) < 2:
        return None
    
    v1 = example['version_data'][0]
    v2 = example['version_data'][1]
    
    # Code stats
    code1 = v1['code'].split('\n')
    code2 = v2['code'].split('\n')
    
    # Docstring stats
    doc1 = v1['docstring'].split('\n')
    doc2 = v2['docstring'].split('\n')
    
    # Calculate diff stats
    code_diff = list(difflib.unified_diff(code1, code2, lineterm=''))
    doc_diff = list(difflib.unified_diff(doc1, doc2, lineterm=''))
    
    # Filter header lines from diff
    code_diff = [line for line in code_diff if not line.startswith('---') and not line.startswith('+++')]
    doc_diff = [line for line in doc_diff if not line.startswith('---') and not line.startswith('+++')]
    
    # Count additions/deletions
    code_additions = len([line for line in code_diff if line.startswith('+')])
    code_deletions = len([line for line in code_diff if line.startswith('-')])
    doc_additions = len([line for line in doc_diff if line.startswith('+')])
    doc_deletions = len([line for line in doc_diff if line.startswith('-')])
    
    # Calculate lengths
    code1_len = len(v1['code'])
    code2_len = len(v2['code'])
    doc1_len = len(v1['docstring'])
    doc2_len = len(v2['docstring'])
    
    # Extract change patterns
    doc_change_ratio = (doc_additions + doc_deletions) / max(1, len(doc1) + len(doc2))
    code_change_ratio = (code_additions + code_deletions) / max(1, len(code1) + len(code2))
    
    # Check if docstring is just formatted (whitespace changes)
    doc_just_formatted = example.get('whitespace_only_docstring', False)
    
    return {
        'code_additions': code_additions,
        'code_deletions': code_deletions,
        'doc_additions': doc_additions,
        'doc_deletions': doc_deletions,
        'code1_len': code1_len,
        'code2_len': code2_len,
        'doc1_len': doc1_len,
        'doc2_len': doc2_len,
        'doc_change_ratio': doc_change_ratio,
        'code_change_ratio': code_change_ratio,
        'doc_just_formatted': doc_just_formatted,
        'total_code_changes': code_additions + code_deletions,
        'total_doc_changes': doc_additions + doc_deletions
    }


def filter_examples(examples, args):
    """Filter examples based on criteria."""
    filtered = []
    stats = []
    
    for ex in tqdm(examples, desc="Filtering examples"):
        if len(ex.get('version_data', [])) < 2:
            continue
            
        # Compute change statistics
        change_stats = compute_change_stats(ex)
        if not change_stats:
            continue
            
        # Apply filters
        if change_stats['total_code_changes'] < args.min_code_change:
            continue
            
        if change_stats['total_doc_changes'] < args.min_doc_change:
            continue
            
        # Skip examples where only whitespace was changed in the docstring
        if change_stats['doc_just_formatted']:
            continue
            
        # Skip examples with overly long code or docstrings
        if (change_stats['code1_len'] > args.max_length or 
            change_stats['code2_len'] > args.max_length or
            change_stats['doc1_len'] > args.max_length or 
            change_stats['doc2_len'] > args.max_length):
            continue
            
        # Add change stats to the example
        ex['change_stats'] = change_stats
        filtered.append(ex)
        stats.append(change_stats)
    
    return filtered, stats


def create_splits(examples, args):
    """Create train/val/test splits."""
    # Get unique projects
    projects = set(ex.get('project', 'unknown') for ex in examples)
    
    if args.project_aware_split:
        # Group examples by project
        project_examples = defaultdict(list)
        for ex in examples:
            project = ex.get('project', 'unknown')
            project_examples[project].append(ex)
        
        # Assign projects to splits
        random.seed(args.seed)
        project_list = list(projects)
        random.shuffle(project_list)
        
        num_projects = len(project_list)
        train_projects = set(project_list[:int(num_projects * args.train_size)])
        val_projects = set(project_list[int(num_projects * args.train_size):
                                        int(num_projects * (args.train_size + args.val_size))])
        test_projects = set(project_list[int(num_projects * (args.train_size + args.val_size)):])
        
        # Assign examples to splits based on project
        train_examples = []
        val_examples = []
        test_examples = []
        
        for project, exs in project_examples.items():
            if project in train_projects:
                train_examples.extend(exs)
            elif project in val_projects:
                val_examples.extend(exs)
            else:
                test_examples.extend(exs)
    else:
        # Randomly assign examples to splits
        random.seed(args.seed)
        random.shuffle(examples)
        
        n = len(examples)
        train_examples = examples[:int(n * args.train_size)]
        val_examples = examples[int(n * args.train_size):int(n * (args.train_size + args.val_size))]
        test_examples = examples[int(n * (args.train_size + args.val_size)):]
    
    # Balance projects if requested
    if args.balance_projects:
        train_examples = balance_by_project(train_examples)
        val_examples = balance_by_project(val_examples)
        test_examples = balance_by_project(test_examples)
    
    return train_examples, val_examples, test_examples


def balance_by_project(examples):
    """Balance examples across different projects."""
    project_examples = defaultdict(list)
    for ex in examples:
        project = ex.get('project', 'unknown')
        project_examples[project].append(ex)
    
    # Find minimum number of examples per project (with a threshold)
    min_examples = max(10, min(len(exs) for exs in project_examples.values()))
    
    # Cap at 95th percentile to avoid extreme imbalance
    counts = [len(exs) for exs in project_examples.values()]
    cap = int(np.percentile(counts, 95))
    
    # Balance by sampling
    balanced = []
    for project, exs in project_examples.items():
        if len(exs) > cap:
            balanced.extend(random.sample(exs, cap))
        else:
            balanced.extend(exs)
    
    return balanced


def prepare_training_data(examples):
    """Convert examples to the format expected by the training script."""
    prepared = []
    
    for ex in examples:
        if len(ex.get('version_data', [])) < 2:
            continue
            
        v1 = ex['version_data'][0]
        v2 = ex['version_data'][1]
        
        # Create instruction format
        instruction = f"""Improve the following Python documentation to align with the updated code:

Original code:
```python
{v1['code']}```

Updated code:
```python
{v2['code']}
```

Original documentation:
```python
{v1['docstring']}
```

Please provide an improved version of the documentation that reflects the code changes."""
        
        # Use the improved docstring as the response
        response = v2['docstring']
        
        prepared.append({
            "instruction": instruction,
            "response": response,
            "file": ex.get('file', ''),
            "function": ex.get('function', ''),
            "project": ex.get('project', 'unknown'),
            "stats": ex.get('change_stats', {})
        })
    
    return prepared


def save_jsonl(data, file_path):
    """Save a list of dictionaries as a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def print_dataset_stats(train, val, test, stats):
    """Print dataset statistics."""
    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {len(train) + len(val) + len(test)}")
    print(f"Train examples: {len(train)}")
    print(f"Validation examples: {len(val)}")
    print(f"Test examples: {len(test)}")
    
    # Project distribution
    train_projects = Counter(ex.get('project', 'unknown') for ex in train)
    val_projects = Counter(ex.get('project', 'unknown') for ex in val)
    test_projects = Counter(ex.get('project', 'unknown') for ex in test)
    
    print("\n=== Project Distribution ===")
    print(f"Unique projects in train: {len(train_projects)}")
    print(f"Unique projects in val: {len(val_projects)}")
    print(f"Unique projects in test: {len(test_projects)}")
    
    # Code and docstring change statistics
    if stats:
        print("\n=== Change Statistics ===")
        code_changes = [stat['total_code_changes'] for stat in stats]
        doc_changes = [stat['total_doc_changes'] for stat in stats]
        
        print(f"Avg code changes per example: {np.mean(code_changes):.2f}")
        print(f"Avg docstring changes per example: {np.mean(doc_changes):.2f}")
        print(f"Max code changes: {max(code_changes)}")
        print(f"Max docstring changes: {max(doc_changes)}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}")
    examples = load_jsonl(args.input)
    print(f"Loaded {len(examples)} examples")
    
    # Filter examples
    filtered_examples, stats = filter_examples(examples, args)
    print(f"Filtered to {len(filtered_examples)} examples")
    
    # Create splits
    train_examples, val_examples, test_examples = create_splits(filtered_examples, args)
    
    # Prepare data for training
    train_data = prepare_training_data(train_examples)
    val_data = prepare_training_data(val_examples)
    test_data = prepare_training_data(test_examples)
    
    # Save processed data
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)
    
    print(f"Saved processed data to {args.output_dir}")
    
    # Print statistics
    print_dataset_stats(train_data, val_data, test_data, stats)
    
    # Save config
    config = vars(args)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved preprocessing config to {config_path}")


if __name__ == "__main__":
    main()