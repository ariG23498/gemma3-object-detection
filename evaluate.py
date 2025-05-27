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
    """
    Parse the model output to extract bounding boxes and categories.
    
    Args:
        output_text: Raw model output text
        width: Image width
        height: Image height
        
    Returns:
        List of dictionaries containing bbox coordinates and category
    """
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
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
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


def calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Calculate Average Precision using the 11-point interpolation method.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        
    Returns:
        Average Precision value
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    # Use 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # Find precisions for recalls >= t
        valid_precisions = sorted_precisions[sorted_recalls >= t]
        if len(valid_precisions) > 0:
            ap += np.max(valid_precisions)
        
    return ap / 11.0


def evaluate_detections(predictions: List[Dict], ground_truths: List[Dict], 
                       iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate detections for a single image.
    
    Args:
        predictions: List of predicted detections
        ground_truths: List of ground truth detections
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Dictionary containing TP, FP, FN counts
    """
    true_positives = 0
    false_positives = 0
    matched_gt = set()
    
    # For each prediction, find the best matching ground truth
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
                
            # Only match if categories are the same
            if pred['category'] != gt['category']:
                continue
                
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    false_negatives = len(ground_truths) - len(matched_gt)
    
    return {
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def compute_metrics(all_results: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Compute overall precision, recall, and F1 score.
    
    Args:
        all_results: List of dictionaries containing TP, FP, FN for each image
        
    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    total_tp = sum(result['tp'] for result in all_results)
    total_fp = sum(result['fp'] for result in all_results)
    total_fn = sum(result['fn'] for result in all_results)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }


def get_dataloader(processor, cfg):
    """Create test dataloader"""
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, collate_fn=test_collate_fn  # Batch size 1 for evaluation
    )
    return test_dataloader, test_dataset


def main():
    """Main evaluation function"""
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
    
    all_results = []
    all_ious = []
    category_results = defaultdict(list)
    
    logger.info(f"Evaluating on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for idx, (sample, sample_images) in enumerate(test_dataloader):
            if idx >= len(test_dataset):
                break
                
            sample = sample.to(cfg.device)
            
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
                
                # Evaluate detections
                result = evaluate_detections(predictions, ground_truths)
                all_results.append(result)
                
                # Calculate IoUs for matched detections
                for pred in predictions:
                    for gt in ground_truths:
                        if pred['category'] == gt['category']:
                            iou = calculate_iou(pred['bbox'], gt['bbox'])
                            all_ious.append(iou)
                            category_results[pred['category']].append(iou)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(test_dataset)} samples")
    
    # Compute overall metrics
    logger.info("Computing final metrics...")
    metrics = compute_metrics(all_results)
    
    # Compute mAP (simplified version)
    # For a more accurate mAP, we would need confidence scores and multiple IoU thresholds
    map_50 = metrics['precision']  # Simplified mAP@0.5
    
    # Compute average IoU
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("OBJECT DETECTION EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {cfg.dataset_id}")
    print(f"Model: {cfg.checkpoint_id}")
    print(f"Test samples: {len(test_dataset)}")
    print("-"*60)
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"mAP@0.5:       {map_50:.4f}")
    print(f"Average IoU:   {avg_iou:.4f}")
    print("-"*60)
    print(f"True Positives:  {metrics['total_tp']}")
    print(f"False Positives: {metrics['total_fp']}")
    print(f"False Negatives: {metrics['total_fn']}")
    print("="*60)
    
    # Category-wise IoU
    if category_results:
        print("\nCategory-wise Average IoU:")
        for category, ious in category_results.items():
            avg_cat_iou = np.mean(ious)
            print(f"  {category}: {avg_cat_iou:.4f}")
    
    return {
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'map_50': map_50,
        'avg_iou': avg_iou,
        'detailed_metrics': metrics
    }


if __name__ == "__main__":
    results = main() 