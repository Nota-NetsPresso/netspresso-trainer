"""
This code is a modified version of https://github.com/roboflow/supervision/blob/a7edf467172df921608f0360112ba70e2259077c/supervision/metrics/detection.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..base import BaseMetric


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def match_detection_batch(
    predictions: np.ndarray, targets: np.ndarray, iou_thresholds: np.ndarray
) -> np.ndarray:
    """
    Match predictions with target labels based on IoU levels.

    Args:
        predictions (np.ndarray): Batch prediction. Describes a single image and
            has `shape = (M, 6)` where `M` is the number of detected objects.
            Each row is expected to be in
            `(x_min, y_min, x_max, y_max, class, conf)` format.
        targets (np.ndarray): Batch target labels. Describes a single image and
            has `shape = (N, 5)` where `N` is the number of ground-truth objects.
            Each row is expected to be in
            `(x_min, y_min, x_max, y_max, class)` format.
        iou_thresholds (np.ndarray): Array contains different IoU thresholds.

    Returns:
        np.ndarray: Matched prediction with target labels result.
    """
    num_predictions, num_iou_levels = predictions.shape[0], iou_thresholds.shape[0]
    correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
    iou = box_iou_batch(targets[:, :4], predictions[:, :4])
    correct_class = targets[:, 4:5] == predictions[:, 4]

    for i, iou_level in enumerate(iou_thresholds):
        matched_indices = np.where((iou >= iou_level) & correct_class)

        if matched_indices[0].shape[0]:
            combined_indices = np.stack(matched_indices, axis=1)
            iou_values = iou[matched_indices][:, None]
            matches = np.hstack([combined_indices, iou_values])

            if matched_indices[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            correct[matches[:, 1].astype(int), i] = True

    return correct


def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute the average precision using 101-point interpolation (COCO), given
        the recall and precision curves.

    Args:
        recall (np.ndarray): The recall curve.
        precision (np.ndarray): The precision curve.

    Returns:
        float: Average precision.
    """
    extended_recall = np.concatenate(([0.0], recall, [1.0]))
    extended_precision = np.concatenate(([1.0], precision, [0.0]))
    max_accumulated_precision = np.flip(
        np.maximum.accumulate(np.flip(extended_precision))
    )
    interpolated_recall_levels = np.linspace(0, 1, 101)
    interpolated_precision = np.interp(
        interpolated_recall_levels, extended_recall, max_accumulated_precision
    )
    average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
    return average_precision


def average_precisions_per_class(
    matches: np.ndarray,
    prediction_confidence: np.ndarray,
    prediction_class_ids: np.ndarray,
    true_class_ids: np.ndarray,
    eps: float = 1e-16,
) -> np.ndarray:
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Args:
        matches (np.ndarray): True positives.
        prediction_confidence (np.ndarray): Objectness value from 0-1.
        prediction_class_ids (np.ndarray): Predicted object classes.
        true_class_ids (np.ndarray): True object classes.
        eps (float, optional): Small value to prevent division by zero.

    Returns:
        np.ndarray: Average precision for different IoU levels.
    """
    sorted_indices = np.argsort(-prediction_confidence)
    matches = matches[sorted_indices]
    prediction_class_ids = prediction_class_ids[sorted_indices]

    unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
    num_classes = unique_classes.shape[0]

    average_precisions = np.zeros((num_classes, matches.shape[1]))

    for class_idx, class_id in enumerate(unique_classes):
        is_class = prediction_class_ids == class_id
        total_true = class_counts[class_idx]
        total_prediction = is_class.sum()

        if total_prediction == 0 or total_true == 0:
            continue

        false_positives = (1 - matches[is_class]).cumsum(0)
        true_positives = matches[is_class].cumsum(0)
        recall = true_positives / (total_true + eps)
        precision = true_positives / (true_positives + false_positives)

        for iou_level_idx in range(matches.shape[1]):
            average_precisions[
                class_idx, iou_level_idx
            ] = compute_average_precision(
                recall[:, iou_level_idx], precision[:, iou_level_idx]
            )

    return average_precisions


class DetectionMetric(BaseMetric):
    metric_names: List[str] = ['map50', 'map75', 'map50_95']
    primary_metric: str = 'map50'

    def __init__(self, **kwargs):
        super().__init__()

    def calibrate(self, predictions, targets, **kwargs):
        result_dict = {k: 0. for k in self.metric_names}

        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        # Gather matching stats for predictions and targets
        for pred, target in zip(predictions, targets):
            predicted_objs_bbox, predicted_objs_class, predicted_objs_confidence = pred['post_boxes'], pred['post_labels'], pred['post_scores']
            true_objs_bbox, true_objs_class = target['boxes'], target['labels']

            true_objs = np.concatenate((true_objs_bbox, true_objs_class[..., np.newaxis]), axis=-1)
            predicted_objs = np.concatenate((predicted_objs_bbox, predicted_objs_class[..., np.newaxis], predicted_objs_confidence[..., np.newaxis]), axis=-1)

            if predicted_objs.shape[0] == 0 and true_objs.shape[0]:
                stats.append(
                    (
                        np.zeros((0, iou_thresholds.size), dtype=bool),
                        *np.zeros((2, 0)),
                        true_objs[:, 4],
                    )
                )

            if true_objs.shape[0]:
                matches = match_detection_batch(predicted_objs, true_objs, iou_thresholds)
                stats.append(
                    (
                        matches,
                        predicted_objs[:, 5],
                        predicted_objs[:, 4],
                        true_objs[:, 4],
                    )
                )

        # Compute average precisions if any matches exist
        if stats:
            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions = average_precisions_per_class(*concatenated_stats)
            result_dict['map50'] = average_precisions[:, 0].mean()
            result_dict['map75'] = average_precisions[:, 5].mean()
            result_dict['map50_95'] = average_precisions.mean()
        else:
            result_dict['map50'], result_dict['map75'], result_dict['map50_95'] = 0, 0, 0
            average_precisions = []

        return result_dict
