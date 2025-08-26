# evaluation.py

import ast
from pathlib import Path
from shapely.geometry import Polygon
from utils import load_from_json

def iou(box_a: list, box_b: list) -> float:
    """Calcula a Intersection over Union (IoU) para bounding boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def iou_segmentation(mask_a: list, mask_b: list) -> float:
    """Calcula a IoU para máscaras de segmentação."""
    try:
        poly1 = Polygon(mask_a)
        poly2 = Polygon(mask_b)
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
    except Exception as e:
        print(f"Erro ao calcular IoU de segmentação: {e}")
        return 0.0

def _add_detection_result(results_dict: dict, class_id: float, confidence: float, label: str):
    """Adiciona um resultado de detecção (TP, FP, FN) ao dicionário."""
    if class_id not in results_dict:
        results_dict[class_id] = [[], []]
    
    results_dict[class_id][0].append(confidence)
    results_dict[class_id][1].append(label)

def _evaluate_frame(results_dict, gt_data, pred_data, iou_threshold, iou_func):
    """Avalia um único frame, calculando TPs, FPs e FNs."""
    gt_classes, gt_confidences = gt_data[0]
    gt_boxes = gt_data[1]
    
    pred_classes, pred_confidences = pred_data[0]
    pred_boxes = pred_data[1]

    matched_gt_indices = set()

    # Calcula TPs e FPs
    for i, (pred_cls, pred_box, pred_conf) in enumerate(zip(pred_classes, pred_boxes, pred_confidences)):
        best_iou = 0
        match_index = -1
        for j, (gt_cls, gt_box) in enumerate(zip(gt_classes, gt_boxes)):
            if j in matched_gt_indices or pred_cls != gt_cls:
                continue
            
            current_iou = iou_func(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                match_index = j

        if best_iou >= iou_threshold:
            _add_detection_result(results_dict, pred_cls, pred_conf, 'TP')
            matched_gt_indices.add(match_index)
        else:
            _add_detection_result(results_dict, pred_cls, pred_conf, 'FP')

    # Calcula FNs
    for i, gt_cls in enumerate(gt_classes):
        if i not in matched_gt_indices:
            _add_detection_result(results_dict, gt_cls, None, 'FN')

def build_results_dictionary(gt_dir: str, pred_dir: str, iou_threshold: float, is_segmentation: bool) -> dict:
    """Constrói o dicionário de resultados (TP, FP, FN) para um diretório inteiro."""
    gt_files = sorted(list(Path(gt_dir).glob('*.json')))
    pred_files = sorted(list(Path(pred_dir).glob('*.json')))

    if len(gt_files) != len(pred_files):
        raise ValueError("Número de arquivos de ground truth e predições não coincide.")

    results_dict = {}
    iou_func = iou_segmentation if is_segmentation else iou

    for gt_path, pred_path in zip(gt_files, pred_files):
        gt_video_data = load_from_json(gt_path)
        pred_video_data = load_from_json(pred_path)

        for gt_frame, pred_frame in zip(gt_video_data, pred_video_data):
            _evaluate_frame(results_dict, gt_frame, pred_frame, iou_threshold, iou_func)
            
    return results_dict

def calculate_average_precision(confidences: list, labels: list) -> float:
    """Calcula o Average Precision (AP) para uma única classe."""
    detections = sorted(
        [(c if c is not None else -1.0, l) for c, l in zip(confidences, labels)],
        key=lambda x: x[0],
        reverse=True
    )

    total_positives = labels.count('TP') + labels.count('FN')
    if total_positives == 0:
        return 0.0

    acc_tp = 0
    acc_fp = 0
    precision_sum = 0.0

    for _, label in detections:
        if label == 'TP':
            acc_tp += 1
            precision = acc_tp / (acc_tp + acc_fp)
            precision_sum += precision
        elif label == 'FP':
            acc_fp += 1
    
    return precision_sum / total_positives

def calculate_map_from_dict(results_dict: dict) -> tuple[float, dict]:
    """Calcula o mAP a partir do dicionário de resultados."""
    if not results_dict:
        return 0.0, {}

    ap_per_class = {}
    for class_id, (confidences, labels) in results_dict.items():
        ap = calculate_average_precision(confidences, labels)
        ap_per_class[class_id] = ap

    mean_ap = sum(ap_per_class.values()) / len(ap_per_class)
    return mean_ap, ap_per_class