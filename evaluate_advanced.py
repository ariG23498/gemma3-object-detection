import logging
import os
from functools import partial
from typing import List, Tuple, Dict, Any
import numpy as np
from collections import defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import test_collate_function, parse_paligemma_label

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_model_output(output_text: str, width: int, height: int) -> List[Dict[str, Any]]:
    """Parse the model output to extract bounding boxes and categories."""
    predictions = []
    
    # Split by semicolon to handle multiple detections
    detections = output_text.split(';')
    
    for detection in detections:
        detection = detection.strip()
        if not detection:
            continue
            
        try:
            category, bbox = parse_paligemma_label(detection, width, height)
            predictions.append({
                'bbox': bbox,  # [x1, y1, x2, y2]
                'category': category.strip(),
                'confidence': 1.0  # Model doesn't output confidence scores
            })
        except Exception as e:
            logger.warning(f"Failed to parse detection: {detection}, Error: {e}")
            continue
    
    return predictions


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def convert_coco_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def calculate_ap_interpolated(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Calculate Average Precision using interpolation method (COCO style).
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]
    
    # Add points at recall 0 and 1
    recalls_interp = np.concatenate(([0], sorted_recalls, [1]))
    precisions_interp = np.concatenate(([0], sorted_precisions, [0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions_interp) - 2, -1, -1):
        precisions_interp[i] = max(precisions_interp[i], precisions_interp[i + 1])
    
    # Calculate AP as area under curve
    ap = 0.0
    for i in range(1, len(recalls_interp)):
        ap += (recalls_interp[i] - recalls_interp[i - 1]) * precisions_interp[i]
    
    return ap


def evaluate_at_iou_threshold(all_predictions: List[List[Dict]], 
                             all_ground_truths: List[List[Dict]], 
                             iou_threshold: float,
                             category: str = None) -> Dict[str, float]:
    """
    Evaluate detections at a specific IoU threshold.
    
    Args:
        all_predictions: List of prediction lists for each image
        all_ground_truths: List of ground truth lists for each image
        iou_threshold: IoU threshold for considering a detection as correct
        category: Specific category to evaluate (None for all categories)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    all_scores = []  # confidence scores
    all_matches = []  # whether each detection is TP or FP
    total_gt = 0
    
    for predictions, ground_truths in zip(all_predictions, all_ground_truths):
        # Filter by category if specified
        if category:
            predictions = [p for p in predictions if p['category'] == category]
            ground_truths = [gt for gt in ground_truths if gt['category'] == category]
        
        total_gt += len(ground_truths)
        
        # Sort predictions by confidence (descending)
        predictions_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        matched_gt = set()
        
        for pred in predictions_sorted:
            all_scores.append(pred['confidence'])
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                    
                # Only match if categories are the same (or we're not filtering by category)
                if category is None and pred['category'] != gt['category']:
                    continue
                    
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                all_matches.append(1)  # True Positive
                matched_gt.add(best_gt_idx)
            else:
                all_matches.append(0)  # False Positive
    
    if not all_scores:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Convert to numpy arrays
    scores = np.array(all_scores)
    matches = np.array(all_matches)
    
    # Sort by confidence scores (descending)
    sorted_indices = np.argsort(-scores)
    matches_sorted = matches[sorted_indices]
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(matches_sorted)
    fp_cumsum = np.cumsum(1 - matches_sorted)
    
    # Calculate precision and recall at each point
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    
    # Calculate Average Precision
    ap = calculate_ap_interpolated(precisions, recalls)
    
    # Final precision and recall
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    
    return {
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'total_tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
        'total_fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
        'total_gt': total_gt
    }


def calculate_comprehensive_metrics(all_predictions: List[List[Dict]], 
                                  all_ground_truths: List[List[Dict]]) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics including mAP at multiple IoU thresholds.
    
    Args:
        all_predictions: List of prediction lists for each image
        all_ground_truths: List of ground truth lists for each image
        
    Returns:
        Dictionary containing comprehensive metrics
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, 0.6, ..., 0.95]
    
    # Get all unique categories
    all_categories = set()
    for predictions in all_predictions:
        for pred in predictions:
            all_categories.add(pred['category'])
    for ground_truths in all_ground_truths:
        for gt in ground_truths:
            all_categories.add(gt['category'])
    
    results = {}
    
    # Calculate metrics for each IoU threshold
    ap_per_iou = []
    for iou_thresh in iou_thresholds:
        metrics = evaluate_at_iou_threshold(all_predictions, all_ground_truths, iou_thresh)
        ap_per_iou.append(metrics['ap'])
        
        if iou_thresh == 0.5:  # Store detailed metrics for IoU=0.5
            results['metrics_at_50'] = metrics
        if iou_thresh == 0.75:  # Store detailed metrics for IoU=0.75
            results['metrics_at_75'] = metrics
    
    # Calculate mAP (mean over IoU thresholds)
    results['mAP'] = np.mean(ap_per_iou)
    results['mAP_50'] = ap_per_iou[0]  # AP at IoU=0.5
    results['mAP_75'] = ap_per_iou[5] if len(ap_per_iou) > 5 else 0.0  # AP at IoU=0.75
    
    # Category-wise metrics at IoU=0.5
    category_metrics = {}
    for category in all_categories:
        cat_metrics = evaluate_at_iou_threshold(all_predictions, all_ground_truths, 0.5, category)
        category_metrics[category] = cat_metrics
    
    results['category_metrics'] = category_metrics
    results['iou_thresholds'] = iou_thresholds.tolist()
    results['ap_per_iou'] = ap_per_iou
    
    return results


def get_dataloader(processor, cfg):
    """Create test dataloader"""
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, collate_fn=test_collate_fn
    )
    return test_dataloader, test_dataset


def main():
    """Main evaluation function with comprehensive metrics"""
    cfg = Configuration()
    
    logger.info("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )
    model.eval()
    model.to(cfg.device)
    
    logger.info("Creating test dataloader...")
    test_dataloader, test_dataset = get_dataloader(processor=processor, cfg=cfg)
    
    all_predictions = []
    all_ground_truths = []
    all_ious = []
    
    logger.info(f"Evaluating on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for idx, (sample, sample_images) in enumerate(test_dataloader):
            if idx >= len(test_dataset):
                break
            
            sample = sample.to(cfg.device)
            print(sample)
            # Get predictions from model
            generation = model.generate(**sample, max_new_tokens=100)
            decoded = processor.batch_decode(generation, skip_special_tokens=True)
            
            # Process each sample in the batch (batch size is 1)
            for batch_idx, (output_text, sample_image) in enumerate(zip(decoded, sample_images)):
                image = sample_image[0]
                width, height = image.size
                
                # Get ground truth from original dataset
                gt_sample = test_dataset[idx]
                ground_truths = []
                
                # Convert ground truth bounding boxes
                for bbox, category in zip(gt_sample['objects']['bbox'], gt_sample['objects']['category']):
                    gt_bbox = convert_coco_to_xyxy(bbox)
                    ground_truths.append({
                        'bbox': gt_bbox,
                        'category': 'plate'  # Assuming all objects are plates
                    })
                
                # Parse model predictions
                predictions = parse_model_output(output_text, width, height)
                
                all_predictions.append(predictions)
                all_ground_truths.append(ground_truths)
                
                # Calculate IoUs for all predictions
                for pred in predictions:
                    for gt in ground_truths:
                        if pred['category'] == gt['category']:
                            iou = calculate_iou(pred['bbox'], gt['bbox'])
                            all_ious.append(iou)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_dataset)} samples")
    
    # Calculate comprehensive metrics
    logger.info("Computing comprehensive metrics...")
    results = calculate_comprehensive_metrics(all_predictions, all_ground_truths)
    
    # Calculate average IoU
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Print results
    print("\n" + "="*70)
    print("COMPREHENSIVE OBJECT DETECTION EVALUATION RESULTS")
    print("="*70)
    print(f"Dataset: {cfg.dataset_id}")
    print(f"Model: {cfg.checkpoint_id}")
    print(f"Test samples: {len(test_dataset)}")
    print("-"*70)
    print("COCO-style mAP Metrics:")
    print(f"  mAP (IoU 0.5:0.95): {results['mAP']:.4f}")
    print(f"  mAP@0.5:           {results['mAP_50']:.4f}")
    print(f"  mAP@0.75:          {results['mAP_75']:.4f}")
    print("-"*70)
    
    if 'metrics_at_50' in results:
        metrics_50 = results['metrics_at_50']
        print("Metrics at IoU=0.5:")
        print(f"  Precision:         {metrics_50['precision']:.4f}")
        print(f"  Recall:            {metrics_50['recall']:.4f}")
        print(f"  True Positives:    {metrics_50['total_tp']}")
        print(f"  False Positives:   {metrics_50['total_fp']}")
        print(f"  Total Ground Truth: {metrics_50['total_gt']}")
    
    print(f"\nAverage IoU:         {avg_iou:.4f}")
    print("-"*70)
    
    # Category-wise results
    if results['category_metrics']:
        print("Category-wise AP@0.5:")
        for category, metrics in results['category_metrics'].items():
            print(f"  {category:12}: AP={metrics['ap']:.4f}, "
                  f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    print("="*70)
    
    # Save results to file
    results['avg_iou'] = avg_iou
    results['total_samples'] = len(test_dataset)
    
    return results


if __name__ == "__main__":
    results = main() 