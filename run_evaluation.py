#!/usr/bin/env python3
"""
Evaluation runner script for Gemma 3 Object Detection model.

Usage:
    python run_evaluation.py --mode basic    # Basic evaluation (faster)
    python run_evaluation.py --mode advanced # Comprehensive COCO-style evaluation
    python run_evaluation.py --help          # Show help
"""

import argparse
import sys
import json
import time
from pathlib import Path


def run_basic_evaluation():
    """Run basic evaluation with simple metrics."""
    print("Running basic evaluation...")
    try:
        import evaluate
        results = evaluate.main()
        return results
    except Exception as e:
        print(f"Error running basic evaluation: {e}")
        return None


def run_advanced_evaluation():
    """Run advanced evaluation with comprehensive COCO-style metrics."""
    print("Running advanced evaluation with COCO-style metrics...")
    try:
        import evaluate_advanced
        results = evaluate_advanced.main()
        return results
    except Exception as e:
        print(f"Error running advanced evaluation: {e}")
        return None


def save_results(results, output_file):
    """Save evaluation results to JSON file."""
    if results is None:
        print("No results to save.")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if hasattr(v, 'tolist'):
                    json_results[key][k] = v.tolist()
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3 Object Detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --mode basic
  python run_evaluation.py --mode advanced --output results/eval_results.json
  python run_evaluation.py --mode both --output results/
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['basic', 'advanced', 'both'],
        default='basic',
        help='Evaluation mode: basic (fast), advanced (comprehensive), or both'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file/directory for results (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("GEMMA 3 OBJECT DETECTION EVALUATION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print("-"*60)
    
    start_time = time.time()
    
    if args.mode == 'basic':
        results = run_basic_evaluation()
        if results:
            save_results(results, args.output)
    
    elif args.mode == 'advanced':
        results = run_advanced_evaluation()
        if results:
            save_results(results, args.output)
    
    elif args.mode == 'both':
        output_dir = Path(args.output)
        if output_dir.is_file():
            output_dir = output_dir.parent
        
        print("\n" + "="*60)
        print("RUNNING BASIC EVALUATION")
        print("="*60)
        basic_results = run_basic_evaluation()
        if basic_results:
            save_results(basic_results, output_dir / 'basic_evaluation.json')
        
        print("\n" + "="*60)
        print("RUNNING ADVANCED EVALUATION")
        print("="*60)
        advanced_results = run_advanced_evaluation()
        if advanced_results:
            save_results(advanced_results, output_dir / 'advanced_evaluation.json')
        
        # Create summary comparison
        if basic_results and advanced_results:
            summary = {
                'basic_metrics': {
                    'precision': basic_results.get('precision', 0),
                    'recall': basic_results.get('recall', 0),
                    'f1_score': basic_results.get('f1_score', 0),
                    'avg_iou': basic_results.get('avg_iou', 0)
                },
                'advanced_metrics': {
                    'mAP': advanced_results.get('mAP', 0),
                    'mAP_50': advanced_results.get('mAP_50', 0),
                    'mAP_75': advanced_results.get('mAP_75', 0),
                    'avg_iou': advanced_results.get('avg_iou', 0)
                },
                'evaluation_time': time.time() - start_time
            }
            save_results(summary, output_dir / 'evaluation_summary.json')
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 