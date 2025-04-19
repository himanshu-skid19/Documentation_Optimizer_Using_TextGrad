#!/usr/bin/env python
"""
Dataset Splitter for Documentation Datasets

This script loads JSONL documentation data and splits it into train/validation/test
sets using various stratification strategies to ensure proper representation.
"""

import os
import json
import random
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Set
import difflib
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Split documentation dataset into train/validation/test sets")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file containing documentation data")
    parser.add_argument("--output_dir", type=str, default="processed_dataset", help="Output directory for split datasets")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion of data for testing")
    parser.add_argument("--project_aware", action="store_true", help="Split by project to prevent data leakage")
    parser.add_argument("--balanced", action="store_true", help="Balance dataset by change size/type")
    parser.add_argument("--min_changes", type=int, default=1, help="Minimum number of changes required")
    parser.add_argument("--max_examples_per_project", type=int, default=None, 
                        help="Maximum examples to take from each project (to prevent project dominance)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Handle both complete JSON objects and partial records
                if line.endswith(','):
                    line = line[:-1]
                if line.startswith('{') and line.endswith('}'):
                    data.append(json.loads(line))
                else:
                    # Try to find a complete JSON object
                    match = re.search(r'(\{.*\})', line)
                    if match:
                        data.append(json.loads(match.group(1)))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}\nLine: {line[:100]}...")
    
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def compute_change_statistics(example):
    """Compute statistics about the changes in an example."""
    if len(example.get('version_data', [])) < 2:
        return None
    
    v1 = example['version_data'][0]
    v2 = example['version_data'][1]
    
    # Extract code and docstring
    code1 = v1.get('code', '')
    code2 = v2.get('code', '')
    doc1 = v1.get('docstring', '')
    doc2 = v2.get('docstring', '')
    
    # Count lines
    code1_lines = code1.split('\n')
    code2_lines = code2.split('\n')
    doc1_lines = doc1.split('\n')
    doc2_lines = doc2.split('\n')
    
    # Calculate diffs
    code_diff = list(difflib.unified_diff(code1_lines, code2_lines, n=0))
    doc_diff = list(difflib.unified_diff(doc1_lines, doc2_lines, n=0))
    
    # Count changed lines
    code_changes = len([line for line in code_diff if line.startswith('+') or line.startswith('-')])
    doc_changes = len([line for line in doc_diff if line.startswith('+') or line.startswith('-')])
    
    # Check if whitespace-only changes
    whitespace_only_code = example.get('whitespace_only_code', False)
    whitespace_only_doc = example.get('whitespace_only_docstring', False)
    
    return {
        'code_changes': code_changes,
        'doc_changes': doc_changes,
        'whitespace_only_code': whitespace_only_code,
        'whitespace_only_doc': whitespace_only_doc,
        'total_changes': code_changes + doc_changes,
        'project': example.get('project', 'unknown'),
        'file': example.get('file', 'unknown'),
    }


def analyze_dataset(data):
    """Analyze the dataset to understand its composition."""
    stats = {
        'total_examples': len(data),
        'valid_examples': 0,
        'projects': Counter(),
        'change_sizes': {
            'small': 0,  # 1-5 changes
            'medium': 0, # 6-20 changes
            'large': 0   # 21+ changes
        },
        'change_types': {
            'code_only': 0,
            'doc_only': 0,
            'both': 0
        }
    }
    
    valid_examples = []
    
    for example in tqdm(data, desc="Analyzing dataset"):
        change_stats = compute_change_statistics(example)
        if not change_stats:
            continue
        
        # Add change stats to example
        example['change_stats'] = change_stats
        valid_examples.append(example)
        
        # Update statistics
        stats['valid_examples'] += 1
        stats['projects'][change_stats['project']] += 1
        
        # Categorize by change size
        total_changes = change_stats['total_changes']
        if total_changes <= 5:
            stats['change_sizes']['small'] += 1
        elif total_changes <= 20:
            stats['change_sizes']['medium'] += 1
        else:
            stats['change_sizes']['large'] += 1
        
        # Categorize by change type
        if change_stats['code_changes'] > 0 and change_stats['doc_changes'] > 0:
            stats['change_types']['both'] += 1
        elif change_stats['code_changes'] > 0:
            stats['change_types']['code_only'] += 1
        elif change_stats['doc_changes'] > 0:
            stats['change_types']['doc_only'] += 1
    
    return valid_examples, stats


def filter_examples(examples, args):
    """Filter examples based on criteria."""
    filtered = []
    
    for ex in examples:
        change_stats = ex.get('change_stats', {})
        
        # Skip examples with too few changes
        if change_stats.get('total_changes', 0) < args.min_changes:
            continue
        
        # Skip examples with only whitespace changes
        if change_stats.get('whitespace_only_code', False) and change_stats.get('whitespace_only_doc', False):
            continue
        
        filtered.append(ex)
    
    return filtered


def balance_by_project(examples, max_per_project=None):
    """Balance examples by project."""
    project_examples = defaultdict(list)
    
    # Group by project
    for ex in examples:
        project = ex.get('change_stats', {}).get('project', 'unknown')
        project_examples[project].append(ex)
    
    # Balance by capping per project
    balanced = []
    for project, exs in project_examples.items():
        if max_per_project and len(exs) > max_per_project:
            # Randomly sample from this project
            balanced.extend(random.sample(exs, max_per_project))
        else:
            balanced.extend(exs)
    
    return balanced


def split_by_project(examples, train_ratio, val_ratio, test_ratio):
    """Split examples by project to prevent data leakage."""
    project_examples = defaultdict(list)
    
    # Group by project
    for ex in examples:
        project = ex.get('change_stats', {}).get('project', 'unknown')
        project_examples[project].append(ex)
    
    # Split projects
    projects = list(project_examples.keys())
    random.shuffle(projects)
    
    n_projects = len(projects)
    train_idx = int(n_projects * train_ratio)
    val_idx = int(n_projects * (train_ratio + val_ratio))
    
    train_projects = projects[:train_idx]
    val_projects = projects[train_idx:val_idx]
    test_projects = projects[val_idx:]
    
    # Create splits
    train_examples = []
    val_examples = []
    test_examples = []
    
    for project in train_projects:
        train_examples.extend(project_examples[project])
    
    for project in val_projects:
        val_examples.extend(project_examples[project])
    
    for project in test_projects:
        test_examples.extend(project_examples[project])
    
    return train_examples, val_examples, test_examples


def split_dataset(examples, args):
    """Split the dataset into train/validation/test sets."""
    if args.project_aware:
        # Split by project to prevent data leakage
        train_examples, val_examples, test_examples = split_by_project(
            examples, args.train_ratio, args.val_ratio, args.test_ratio
        )
    else:
        # Random split
        random.shuffle(examples)
        n = len(examples)
        train_idx = int(n * args.train_ratio)
        val_idx = int(n * (args.train_ratio + args.val_ratio))
        
        train_examples = examples[:train_idx]
        val_examples = examples[train_idx:val_idx]
        test_examples = examples[val_idx:]
    
    # Balance splits if requested
    if args.balanced:
        if args.max_examples_per_project:
            train_examples = balance_by_project(train_examples, args.max_examples_per_project)
            val_examples = balance_by_project(val_examples, args.max_examples_per_project)
            test_examples = balance_by_project(test_examples, args.max_examples_per_project)
    
    return train_examples, val_examples, test_examples


def prepare_for_training(examples):
    """
    Prepare examples for training by converting them to the instruction/response format.
    """
    prepared = []
    
    for ex in examples:
        if len(ex.get('version_data', [])) < 2:
            continue
        
        v1 = ex['version_data'][0]
        v2 = ex['version_data'][1]
        
        # Create instruction
        instruction = f"""Improve the following Python documentation to align with the updated code:

Original code:
```python
{v1.get('code', '')}```

Updated code:
```python
{v2.get('code', '')}
```

Original documentation:
```python
{v1.get('docstring', '')}
```

Please provide an improved version of the documentation that reflects the code changes."""
        
        # Use the updated docstring as the response
        response = v2.get('docstring', '')
        
        # Add metadata
        prepared.append({
            "instruction": instruction,
            "response": response,
            "file": ex.get('file', ''),
            "function": ex.get('function', ''),
            "project": ex.get('change_stats', {}).get('project', 'unknown'),
            "change_stats": {
                "code_changes": ex.get('change_stats', {}).get('code_changes', 0),
                "doc_changes": ex.get('change_stats', {}).get('doc_changes', 0),
                "total_changes": ex.get('change_stats', {}).get('total_changes', 0)
            }
        })
    
    return prepared


def print_stats(data, split_name):
    """Print statistics about a dataset split."""
    projects = Counter([ex.get('project', 'unknown') for ex in data])
    changes = [ex.get('change_stats', {}).get('total_changes', 0) for ex in data]
    
    print(f"\n{split_name} Statistics:")
    print(f"  Examples: {len(data)}")
    print(f"  Projects: {len(projects)}")
    print(f"  Top projects: {projects.most_common(3)}")
    
    if changes:
        print(f"  Avg changes per example: {sum(changes)/len(changes):.2f}")
        print(f"  Min changes: {min(changes)}")
        print(f"  Max changes: {max(changes)}")


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} examples")
    
    # Analyze and filter data
    valid_examples, stats = analyze_dataset(data)
    print("\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Valid examples: {stats['valid_examples']}")
    print(f"  Projects: {len(stats['projects'])}")
    print(f"  Change sizes: {stats['change_sizes']}")
    print(f"  Change types: {stats['change_types']}")
    
    # Filter examples
    filtered_examples = filter_examples(valid_examples, args)
    print(f"\nFiltered to {len(filtered_examples)} examples meeting criteria")
    
    # Split dataset
    train_examples, val_examples, test_examples = split_dataset(filtered_examples, args)
    
    # Print statistics about each split
    print_stats(train_examples, "Train")
    print_stats(val_examples, "Validation")
    print_stats(test_examples, "Test")
    
    # Prepare for training
    train_data = prepare_for_training(train_examples)
    val_data = prepare_for_training(val_examples)
    test_data = prepare_for_training(test_examples)
    
    # Save splits
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)
    
    print(f"\nSaved processed data to {args.output_dir}")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")


if __name__ == "__main__":
    main()