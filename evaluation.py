import logging
from functools import partial

from config import Configuration
from utils import parse_paligemma_label, test_collate_function

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def get_dataloader(processor, cfg):
    test_dataset = load_dataset(cfg.dataset_id, split="test")
    test_collate_fn = partial(
        test_collate_function, processor=processor, dtype=cfg.dtype
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn
    )
    return test_dataloader


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
        
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def compute_matches(predictions: list[dict], ground_truths: list[dict], 
                   iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """
    Compute true positives, false positives, and false negatives.
    Returns: (tp, fp, fn)
    """
    gt_matched = [False] * len(ground_truths)
    tp = 0
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx] or pred["category"] != gt["category"]:
                continue
                
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
    
    fp = len(predictions) - tp
    fn = len(ground_truths) - sum(gt_matched)
    
    return tp, fp, fn


def compute_ap(all_predictions: list[list[dict]], 
               all_ground_truths: list[list[dict]], 
               iou_threshold: float) -> float:
    """Compute Average Precision (AP) for given IoU threshold."""
    # Collect all predictions with image indices
    all_pred_with_img_idx = []
    for img_idx, predictions in enumerate(all_predictions):
        for pred in predictions:
            all_pred_with_img_idx.append((img_idx, pred))
    
    # Track matched ground truths
    gt_matched = [set() for _ in range(len(all_ground_truths))]
    
    tp_list = []
    fp_list = []
    
    for img_idx, pred in all_pred_with_img_idx:
        ground_truths = all_ground_truths[img_idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in gt_matched[img_idx]:
                continue
                
            if pred["category"] != gt["category"]:
                continue
                
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[img_idx].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    total_gt = sum(len(gt) for gt in all_ground_truths)
    
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for recall_level in np.linspace(0, 1, 11):
        matching_recalls = recalls >= recall_level
        if np.any(matching_recalls):
            ap += np.max(precisions[matching_recalls])
    
    return ap / 11.0


def compute_metrics(all_predictions: list[list[dict]], 
                   all_ground_truths: list[list[dict]]) -> dict[str, float]:
    """Compute precision, recall, F1, and mAP metrics."""
    # Basic metrics at IoU 0.5
    total_tp = total_fp = total_fn = 0
    
    for predictions, ground_truths in zip(all_predictions, all_ground_truths):
        tp, fp, fn = compute_matches(predictions, ground_truths, 0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute mAP at different IoU thresholds
    ap_50 = compute_ap(all_predictions, all_ground_truths, 0.5)
    ap_75 = compute_ap(all_predictions, all_ground_truths, 0.75)
    ap_90 = compute_ap(all_predictions, all_ground_truths, 0.9)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP@0.5": ap_50,
        "mAP@0.75": ap_75,  
        "mAP@0.9": ap_90,
        "total_predictions": total_tp + total_fp,
        "total_ground_truths": total_tp + total_fn
    }


def run_evaluation():
    """Main evaluation function"""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    cfg = Configuration()
    
    # Load model and processor
    logger.info("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(cfg.checkpoint_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.checkpoint_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
    )
    model.eval()
    model.to(cfg.device)

    test_dataloader = get_dataloader(processor, cfg)
    
    logger.info("Starting evaluation...")
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for batch_idx, (batch, batch_images) in enumerate(test_dataloader):
            batch = batch.to(cfg.device)
            
            # Generate predictions
            generation = model.generate(**batch, max_new_tokens=100, do_sample=False)
            decoded = processor.batch_decode(generation, skip_special_tokens=True)
            
            # Process each sample in batch
            for sample_idx, (output_text, sample_image) in enumerate(zip(decoded, batch_images)):
                image = sample_image[0]
                width, height = image.size
                
                # Get original sample for ground truth
                original_sample = test_dataloader.dataset[batch_idx * test_dataloader.batch_size + sample_idx]
                
                # Parse predictions
                predictions = []
                prediction_text = output_text
                if "detect" in prediction_text:
                    prediction_text = prediction_text.split("detect")[-1].strip()
                
                for detection in prediction_text.split(';'):
                    detection = detection.strip()
                    if not detection:
                        continue
                        
                    try:
                        category, bbox = parse_paligemma_label(detection, width, height)
                        predictions.append({
                            "bbox": bbox,
                            "category": category.strip()
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse prediction: {detection}")
                        continue
                
                all_predictions.append(predictions)

                # Parse ground truths
                gt_boxes = []
                width, height = original_sample["width"], original_sample["height"]
                
                if "label_for_paligemma" in original_sample:
                    for detection in original_sample["label_for_paligemma"].split(';'):
                        detection = detection.strip()
                        if not detection:
                            continue
                            
                        try:
                            category, bbox = parse_paligemma_label(detection, width, height)
                            gt_boxes.append({
                                "bbox": bbox,
                                "category": category.strip()
                            })
                        except Exception as e:
                            logger.warning(f"Failed to parse ground truth: {detection}")
                            continue

                all_ground_truths.append(gt_boxes)

    # Compute metrics
    results = compute_metrics(all_predictions, all_ground_truths)
    
    # Print results
    logger.info("\n" + "="*40)
    logger.info("EVALUATION RESULTS")
    logger.info("="*40)
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")
    logger.info(f"F1-Score:  {results['f1_score']:.4f}")
    logger.info(f"mAP@0.5:   {results['mAP@0.5']:.4f}")
    logger.info(f"mAP@0.75:  {results['mAP@0.75']:.4f}")
    logger.info(f"mAP@0.9:   {results['mAP@0.9']:.4f}")
    logger.info(f"Total Predictions:   {results['total_predictions']}")
    logger.info(f"Total Ground Truths: {results['total_ground_truths']}")
    
    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write("OBJECT DETECTION EVALUATION RESULTS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n")
        f.write(f"mAP@0.5:   {results['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.75:  {results['mAP@0.75']:.4f}\n")
        f.write(f"mAP@0.9:   {results['mAP@0.9']:.4f}\n")
        f.write(f"Total Predictions:   {results['total_predictions']}\n")
        f.write(f"Total Ground Truths: {results['total_ground_truths']}\n")
    
    logger.info("Results saved to evaluation_results.txt")
    return results


if __name__ == "__main__":
    run_evaluation()