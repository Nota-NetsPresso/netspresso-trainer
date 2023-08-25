"""
This code is a modified version of https://github.com/roboflow/supervision/blob/a7edf467172df921608f0360112ba70e2259077c/supervision/metrics/detection.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..base import BaseMetric

# from supervision.dataset.core import DetectionDataset
# from supervision.detection.core import Detections
# from supervision.detection.utils import box_iou_batch


# def detections_to_tensor(
#     detections: Detections, with_confidence: bool = False
# ) -> np.ndarray:
#     """
#     Convert Supervision Detections to numpy tensors for further computation
#     Args:
#         detections (sv.Detections): Detections/Targets in the format of sv.Detections
#         with_confidence (bool): Whether to include confidence in the tensor
#     Returns:
#         (np.ndarray): Detections as numpy tensors as in (xyxy, class_id,
#             confidence) order
#     """
#     if detections.class_id is None:
#         raise ValueError(
#             "ConfusionMatrix can only be calculated for Detections with class_id"
#         )

#     arrays_to_concat = [detections.xyxy, np.expand_dims(detections.class_id, 1)]

#     if with_confidence:
#         if detections.confidence is None:
#             raise ValueError(
#                 "ConfusionMatrix can only be calculated for Detections with confidence"
#             )
#         arrays_to_concat.append(np.expand_dims(detections.confidence, 1))

#     return np.concatenate(arrays_to_concat, axis=1)


# def validate_input_tensors(predictions: List[np.ndarray], targets: List[np.ndarray]):
#     """
#     Checks for shape consistency of input tensors.
#     """
#     if len(predictions) != len(targets):
#         raise ValueError(
#             f"Number of predictions ({len(predictions)}) and"
#             f"targets ({len(targets)}) must be equal."
#         )
#     if len(predictions) > 0:
#         if not isinstance(predictions[0], np.ndarray) or not isinstance(
#             targets[0], np.ndarray
#         ):
#             raise ValueError(
#                 f"Predictions and targets must be lists of numpy arrays."
#                 f"Got {type(predictions[0])} and {type(targets[0])} instead."
#             )
#         if predictions[0].shape[1] != 6:
#             raise ValueError(
#                 f"Predictions must have shape (N, 6)."
#                 f"Got {predictions[0].shape} instead."
#             )
#         if targets[0].shape[1] != 5:
#             raise ValueError(
#                 f"Targets must have shape (N, 5). Got {targets[0].shape} instead."
#             )


# @dataclass
# class ConfusionMatrix:
#     """
#     Confusion matrix for object detection tasks.

#     Attributes:
#         matrix (np.ndarray): An 2D `np.ndarray` of shape
#             `(len(classes) + 1, len(classes) + 1)`
#             containing the number of `TP`, `FP`, `FN` and `TN` for each class.
#         classes (List[str]): Model class names.
#         conf_threshold (float): Detection confidence threshold between `0` and `1`.
#             Detections with lower confidence will be excluded from the matrix.
#         iou_threshold (float): Detection IoU threshold between `0` and `1`.
#             Detections with lower IoU will be classified as `FP`.
#     """

#     matrix: np.ndarray
#     classes: List[str]
#     conf_threshold: float
#     iou_threshold: float

#     @classmethod
#     def from_detections(
#         cls,
#         predictions: List[Detections],
#         targets: List[Detections],
#         classes: List[str],
#         conf_threshold: float = 0.3,
#         iou_threshold: float = 0.5,
#     ) -> ConfusionMatrix:
#         """
#         Calculate confusion matrix based on predicted and ground-truth detections.

#         Args:
#             targets (List[Detections]): Detections objects from ground-truth.
#             predictions (List[Detections]): Detections objects predicted by the model.
#             classes (List[str]): Model class names.
#             conf_threshold (float): Detection confidence threshold between `0` and `1`.
#                 Detections with lower confidence will be excluded.
#             iou_threshold (float): Detection IoU threshold between `0` and `1`.
#                 Detections with lower IoU will be classified as `FP`.

#         Returns:
#             ConfusionMatrix: New instance of ConfusionMatrix.

#         Example:
#             ```python
#             >>> import supervision as sv

#             >>> targets = [
#             ...     sv.Detections(...),
#             ...     sv.Detections(...)
#             ... ]

#             >>> predictions = [
#             ...     sv.Detections(...),
#             ...     sv.Detections(...)
#             ... ]

#             >>> confusion_matrix = sv.ConfusionMatrix.from_detections(
#             ...     predictions=predictions,
#             ...     targets=target,
#             ...     classes=['person', ...]
#             ... )

#             >>> confusion_matrix.matrix
#             array([
#                 [0., 0., 0., 0.],
#                 [0., 1., 0., 1.],
#                 [0., 1., 1., 0.],
#                 [1., 1., 0., 0.]
#             ])
#             ```
#         """

#         prediction_tensors = []
#         target_tensors = []
#         for prediction, target in zip(predictions, targets):
#             prediction_tensors.append(
#                 detections_to_tensor(prediction, with_confidence=True)
#             )
#             target_tensors.append(detections_to_tensor(target, with_confidence=False))
#         return cls.from_tensors(
#             predictions=prediction_tensors,
#             targets=target_tensors,
#             classes=classes,
#             conf_threshold=conf_threshold,
#             iou_threshold=iou_threshold,
#         )

#     @classmethod
#     def from_tensors(
#         cls,
#         predictions: List[np.ndarray],
#         targets: List[np.ndarray],
#         classes: List[str],
#         conf_threshold: float = 0.3,
#         iou_threshold: float = 0.5,
#     ) -> ConfusionMatrix:
#         """
#         Calculate confusion matrix based on predicted and ground-truth detections.

#         Args:
#             predictions (List[np.ndarray]): Each element of the list describes a single
#                 image and has `shape = (M, 6)` where `M` is the number of detected
#                 objects. Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class, conf)` format.
#             targets (List[np.ndarray]): Each element of the list describes a single
#                 image and has `shape = (N, 5)` where `N` is the number of
#                 ground-truth objects. Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class)` format.
#             classes (List[str]): Model class names.
#             conf_threshold (float): Detection confidence threshold between `0` and `1`.
#                 Detections with lower confidence will be excluded.
#             iou_threshold (float): Detection iou  threshold between `0` and `1`.
#                 Detections with lower iou will be classified as `FP`.

#         Returns:
#             ConfusionMatrix: New instance of ConfusionMatrix.

#         Example:
#             ```python
#             >>> import supervision as sv

#             >>> targets = (
#             ...     [
#             ...         array(
#             ...             [
#             ...                 [0.0, 0.0, 3.0, 3.0, 1],
#             ...                 [2.0, 2.0, 5.0, 5.0, 1],
#             ...                 [6.0, 1.0, 8.0, 3.0, 2],
#             ...             ]
#             ...         ),
#             ...         array([1.0, 1.0, 2.0, 2.0, 2]),
#             ...     ]
#             ... )

#             >>> predictions = [
#             ...     array(
#             ...         [
#             ...             [0.0, 0.0, 3.0, 3.0, 1, 0.9],
#             ...             [0.1, 0.1, 3.0, 3.0, 0, 0.9],
#             ...             [6.0, 1.0, 8.0, 3.0, 1, 0.8],
#             ...             [1.0, 6.0, 2.0, 7.0, 1, 0.8],
#             ...         ]
#             ...     ),
#             ...     array([[1.0, 1.0, 2.0, 2.0, 2, 0.8]])
#             ... ]

#             >>> confusion_matrix = sv.ConfusionMatrix.from_tensors(
#             ...     predictions=predictions,
#             ...     targets=targets,
#             ...     classes=['person', ...]
#             ... )

#             >>> confusion_matrix.matrix
#             array([
#                 [0., 0., 0., 0.],
#                 [0., 1., 0., 1.],
#                 [0., 1., 1., 0.],
#                 [1., 1., 0., 0.]
#             ])
#             ```
#         """
#         validate_input_tensors(predictions, targets)

#         num_classes = len(classes)
#         matrix = np.zeros((num_classes + 1, num_classes + 1))
#         for true_batch, detection_batch in zip(targets, predictions):
#             matrix += cls.evaluate_detection_batch(
#                 predictions=detection_batch,
#                 targets=true_batch,
#                 num_classes=num_classes,
#                 conf_threshold=conf_threshold,
#                 iou_threshold=iou_threshold,
#             )
#         return cls(
#             matrix=matrix,
#             classes=classes,
#             conf_threshold=conf_threshold,
#             iou_threshold=iou_threshold,
#         )

#     @staticmethod
#     def evaluate_detection_batch(
#         predictions: np.ndarray,
#         targets: np.ndarray,
#         num_classes: int,
#         conf_threshold: float,
#         iou_threshold: float,
#     ) -> np.ndarray:
#         """
#         Calculate confusion matrix for a batch of detections for a single image.

#         Args:
#             predictions (np.ndarray): Batch prediction. Describes a single image and
#                 has `shape = (M, 6)` where `M` is the number of detected objects.
#                 Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class, conf)` format.
#             targets (np.ndarray): Batch target labels. Describes a single image and
#                 has `shape = (N, 5)` where `N` is the number of ground-truth objects.
#                 Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class)` format.
#             num_classes (int): Number of classes.
#             conf_threshold (float): Detection confidence threshold between `0` and `1`.
#                 Detections with lower confidence will be excluded.
#             iou_threshold (float): Detection iou  threshold between `0` and `1`.
#                 Detections with lower iou will be classified as `FP`.

#         Returns:
#             np.ndarray: Confusion matrix based on a single image.
#         """
#         result_matrix = np.zeros((num_classes + 1, num_classes + 1))

#         conf_idx = 5
#         confidence = predictions[:, conf_idx]
#         detection_batch_filtered = predictions[confidence > conf_threshold]

#         class_id_idx = 4
#         true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
#         detection_classes = np.array(
#             detection_batch_filtered[:, class_id_idx], dtype=np.int16
#         )
#         true_boxes = targets[:, :class_id_idx]
#         detection_boxes = detection_batch_filtered[:, :class_id_idx]

#         iou_batch = box_iou_batch(
#             boxes_true=true_boxes, boxes_detection=detection_boxes
#         )
#         matched_idx = np.asarray(iou_batch > iou_threshold).nonzero()

#         if matched_idx[0].shape[0]:
#             matches = np.stack(
#                 (matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1
#             )
#             matches = ConfusionMatrix._drop_extra_matches(matches=matches)
#         else:
#             matches = np.zeros((0, 3))

#         matched_true_idx, matched_detection_idx, _ = matches.transpose().astype(
#             np.int16
#         )

#         for i, true_class_value in enumerate(true_classes):
#             j = matched_true_idx == i
#             if matches.shape[0] > 0 and sum(j) == 1:
#                 result_matrix[
#                     true_class_value, detection_classes[matched_detection_idx[j]]
#                 ] += 1  # TP
#             else:
#                 result_matrix[true_class_value, num_classes] += 1  # FN

#         for i, detection_class_value in enumerate(detection_classes):
#             if not any(matched_detection_idx == i):
#                 result_matrix[num_classes, detection_class_value] += 1  # FP

#         return result_matrix

#     @staticmethod
#     def _drop_extra_matches(matches: np.ndarray) -> np.ndarray:
#         """
#         Deduplicate matches. If there are multiple matches for the same true or
#         predicted box, only the one with the highest IoU is kept.
#         """
#         if matches.shape[0] > 0:
#             matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#             matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#         return matches

#     @classmethod
#     def benchmark(
#         cls,
#         dataset: DetectionDataset,
#         callback: Callable[[np.ndarray], Detections],
#         conf_threshold: float = 0.3,
#         iou_threshold: float = 0.5,
#     ) -> ConfusionMatrix:
#         """
#         Calculate confusion matrix from dataset and callback function.

#         Args:
#             dataset (DetectionDataset): Object detection dataset used for evaluation.
#             callback (Callable[[np.ndarray], Detections]): Function that takes an image
#                 as input and returns Detections object.
#             conf_threshold (float): Detection confidence threshold between `0` and `1`.
#                 Detections with lower confidence will be excluded.
#             iou_threshold (float): Detection IoU threshold between `0` and `1`.
#                 Detections with lower IoU will be classified as `FP`.

#         Returns:
#             ConfusionMatrix: New instance of ConfusionMatrix.

#         Example:
#             ```python
#             >>> import supervision as sv
#             >>> from ultralytics import YOLO

#             >>> dataset = sv.DetectionDataset.from_yolo(...)

#             >>> model = YOLO(...)
#             >>> def callback(image: np.ndarray) -> sv.Detections:
#             ...     result = model(image)[0]
#             ...     return sv.Detections.from_ultralytics(result)

#             >>> confusion_matrix = sv.ConfusionMatrix.benchmark(
#             ...     dataset = dataset,
#             ...     callback = callback
#             ... )

#             >>> confusion_matrix.matrix
#             array([
#                 [0., 0., 0., 0.],
#                 [0., 1., 0., 1.],
#                 [0., 1., 1., 0.],
#                 [1., 1., 0., 0.]
#             ])
#             ```
#         """
#         predictions, targets = [], []
#         for img_name, img in dataset.images.items():
#             predictions_batch = callback(img)
#             predictions.append(predictions_batch)
#             targets_batch = dataset.annotations[img_name]
#             targets.append(targets_batch)
#         return cls.from_detections(
#             predictions=predictions,
#             targets=targets,
#             classes=dataset.classes,
#             conf_threshold=conf_threshold,
#             iou_threshold=iou_threshold,
#         )


# @dataclass(frozen=True)
# class MeanAveragePrecision:
#     """
#     Mean Average Precision for object detection tasks.

#     Attributes:
#         map50_95 (float): Mean Average Precision (mAP) calculated over IoU thresholds
#             ranging from `0.50` to `0.95` with a step size of `0.05`.
#         map50 (float): Mean Average Precision (mAP) calculated specifically at
#             an IoU threshold of `0.50`.
#         map75 (float): Mean Average Precision (mAP) calculated specifically at
#             an IoU threshold of `0.75`.
#         per_class_ap50_95 (np.ndarray): Average Precision (AP) values calculated over
#             IoU thresholds ranging from `0.50` to `0.95` with a step size of `0.05`,
#             provided for each individual class.
#     """

#     map50_95: float
#     map50: float
#     map75: float
#     per_class_ap50_95: np.ndarray

#     @classmethod
#     def from_detections(
#         cls,
#         predictions: List[Detections],
#         targets: List[Detections],
#     ) -> MeanAveragePrecision:
#         """
#         Calculate mean average precision based on predicted and ground-truth detections.

#         Args:
#             targets (List[Detections]): Detections objects from ground-truth.
#             predictions (List[Detections]): Detections objects predicted by the model.
#         Returns:
#             MeanAveragePrecision: New instance of ConfusionMatrix.

#         Example:
#             ```python
#             >>> import supervision as sv

#             >>> targets = [
#             ...     sv.Detections(...),
#             ...     sv.Detections(...)
#             ... ]

#             >>> predictions = [
#             ...     sv.Detections(...),
#             ...     sv.Detections(...)
#             ... ]

#             >>> mean_average_precision = sv.MeanAveragePrecision.from_detections(
#             ...     predictions=predictions,
#             ...     targets=target,
#             ... )

#             >>> mean_average_precison.map50_95
#             0.2899
#             ```
#         """
#         prediction_tensors = []
#         target_tensors = []
#         for prediction, target in zip(predictions, targets):
#             prediction_tensors.append(
#                 detections_to_tensor(prediction, with_confidence=True)
#             )
#             target_tensors.append(detections_to_tensor(target, with_confidence=False))
#         return cls.from_tensors(
#             predictions=prediction_tensors,
#             targets=target_tensors,
#         )

#     @classmethod
#     def benchmark(
#         cls,
#         dataset: DetectionDataset,
#         callback: Callable[[np.ndarray], Detections],
#     ) -> MeanAveragePrecision:
#         """
#         Calculate mean average precision from dataset and callback function.

#         Args:
#             dataset (DetectionDataset): Object detection dataset used for evaluation.
#             callback (Callable[[np.ndarray], Detections]): Function that takes
#                 an image as input and returns Detections object.
#         Returns:
#             MeanAveragePrecision: New instance of MeanAveragePrecision.

#         Example:
#             ```python
#             >>> import supervision as sv
#             >>> from ultralytics import YOLO

#             >>> dataset = sv.DetectionDataset.from_yolo(...)

#             >>> model = YOLO(...)
#             >>> def callback(image: np.ndarray) -> sv.Detections:
#             ...     result = model(image)[0]
#             ...     return sv.Detections.from_ultralytics(result)

#             >>> mean_average_precision = sv.MeanAveragePrecision.benchmark(
#             ...     dataset = dataset,
#             ...     callback = callback
#             ... )

#             >>> mean_average_precision.map50_95
#             0.433
#             ```
#         """
#         predictions, targets = [], []
#         for img_name, img in dataset.images.items():
#             predictions_batch = callback(img)
#             predictions.append(predictions_batch)
#             targets_batch = dataset.annotations[img_name]
#             targets.append(targets_batch)
#         return cls.from_detections(
#             predictions=predictions,
#             targets=targets,
#         )

#     @classmethod
#     def from_tensors(
#         cls,
#         predictions: List[np.ndarray],
#         targets: List[np.ndarray],
#     ) -> MeanAveragePrecision:
#         """
#         Calculate Mean Average Precision based on predicted and ground-truth
#             detections at different threshold.

#         Args:
#             predictions (List[np.ndarray]): Each element of the list describes
#                 a single image and has `shape = (M, 6)` where `M` is
#                 the number of detected objects. Each row is expected to be
#                 in `(x_min, y_min, x_max, y_max, class, conf)` format.
#             targets (List[np.ndarray]): Each element of the list describes a single
#                 image and has `shape = (N, 5)` where `N` is the
#                 number of ground-truth objects. Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class)` format.
#         Returns:
#             MeanAveragePrecision: New instance of MeanAveragePrecision.

#         Example:
#             ```python
#             >>> import supervision as sv

#             >>> targets = (
#             ...     [
#             ...         array(
#             ...             [
#             ...                 [0.0, 0.0, 3.0, 3.0, 1],
#             ...                 [2.0, 2.0, 5.0, 5.0, 1],
#             ...                 [6.0, 1.0, 8.0, 3.0, 2],
#             ...             ]
#             ...         ),
#             ...         array([1.0, 1.0, 2.0, 2.0, 2]),
#             ...     ]
#             ... )

#             >>> predictions = [
#             ...     array(
#             ...         [
#             ...             [0.0, 0.0, 3.0, 3.0, 1, 0.9],
#             ...             [0.1, 0.1, 3.0, 3.0, 0, 0.9],
#             ...             [6.0, 1.0, 8.0, 3.0, 1, 0.8],
#             ...             [1.0, 6.0, 2.0, 7.0, 1, 0.8],
#             ...         ]
#             ...     ),
#             ...     array([[1.0, 1.0, 2.0, 2.0, 2, 0.8]])
#             ... ]

#             >>> mean_average_precison = sv.MeanAveragePrecision.from_tensors(
#             ...     predictions=predictions,
#             ...     targets=targets,
#             ... )

#             >>> mean_average_precison.map50_95
#             0.2899
#             ```
#         """
#         validate_input_tensors(predictions, targets)
#         iou_thresholds = np.linspace(0.5, 0.95, 10)
#         stats = []

#         # Gather matching stats for predictions and targets
#         for true_objs, predicted_objs in zip(targets, predictions):
#             if predicted_objs.shape[0] == 0:
#                 if true_objs.shape[0]:
#                     stats.append(
#                         (
#                             np.zeros((0, iou_thresholds.size), dtype=bool),
#                             *np.zeros((2, 0)),
#                             true_objs[:, 4],
#                         )
#                     )
#                 continue

#             if true_objs.shape[0]:
#                 matches = cls._match_detection_batch(
#                     predicted_objs, true_objs, iou_thresholds
#                 )
#                 stats.append(
#                     (
#                         matches,
#                         predicted_objs[:, 5],
#                         predicted_objs[:, 4],
#                         true_objs[:, 4],
#                     )
#                 )

#         # Compute average precisions if any matches exist
#         if stats:
#             concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
#             average_precisions = cls._average_precisions_per_class(*concatenated_stats)
#             map50 = average_precisions[:, 0].mean()
#             map75 = average_precisions[:, 5].mean()
#             map50_95 = average_precisions.mean()
#         else:
#             map50, map75, map50_95 = 0, 0, 0
#             average_precisions = []

#         return cls(
#             map50_95=map50_95,
#             map50=map50,
#             map75=map75,
#             per_class_ap50_95=average_precisions,
#         )

#     @staticmethod
#     def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
#         """
#         Compute the average precision using 101-point interpolation (COCO), given
#             the recall and precision curves.

#         Args:
#             recall (np.ndarray): The recall curve.
#             precision (np.ndarray): The precision curve.

#         Returns:
#             float: Average precision.
#         """
#         extended_recall = np.concatenate(([0.0], recall, [1.0]))
#         extended_precision = np.concatenate(([1.0], precision, [0.0]))
#         max_accumulated_precision = np.flip(
#             np.maximum.accumulate(np.flip(extended_precision))
#         )
#         interpolated_recall_levels = np.linspace(0, 1, 101)
#         interpolated_precision = np.interp(
#             interpolated_recall_levels, extended_recall, max_accumulated_precision
#         )
#         average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
#         return average_precision

#     @staticmethod
#     def _match_detection_batch(
#         predictions: np.ndarray, targets: np.ndarray, iou_thresholds: np.ndarray
#     ) -> np.ndarray:
#         """
#         Match predictions with target labels based on IoU levels.

#         Args:
#             predictions (np.ndarray): Batch prediction. Describes a single image and
#                 has `shape = (M, 6)` where `M` is the number of detected objects.
#                 Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class, conf)` format.
#             targets (np.ndarray): Batch target labels. Describes a single image and
#                 has `shape = (N, 5)` where `N` is the number of ground-truth objects.
#                 Each row is expected to be in
#                 `(x_min, y_min, x_max, y_max, class)` format.
#             iou_thresholds (np.ndarray): Array contains different IoU thresholds.

#         Returns:
#             np.ndarray: Matched prediction with target labels result.
#         """
#         num_predictions, num_iou_levels = predictions.shape[0], iou_thresholds.shape[0]
#         correct = np.zeros((num_predictions, num_iou_levels), dtype=bool)
#         iou = box_iou_batch(targets[:, :4], predictions[:, :4])
#         correct_class = targets[:, 4:5] == predictions[:, 4]

#         for i, iou_level in enumerate(iou_thresholds):
#             matched_indices = np.where((iou >= iou_level) & correct_class)

#             if matched_indices[0].shape[0]:
#                 combined_indices = np.stack(matched_indices, axis=1)
#                 iou_values = iou[matched_indices][:, None]
#                 matches = np.hstack([combined_indices, iou_values])

#                 if matched_indices[0].shape[0] > 1:
#                     matches = matches[matches[:, 2].argsort()[::-1]]
#                     matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                     matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

#                 correct[matches[:, 1].astype(int), i] = True

#         return correct


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


@staticmethod
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


def detection_metrics(pred, target, **kwargs):
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    stats = []

    # Gather matching stats for predictions and targets
    for true_objs, predicted_objs in zip(target, pred):
        if predicted_objs.shape[0] == 0:
            if true_objs.shape[0]:
                stats.append(
                    (
                        np.zeros((0, iou_thresholds.size), dtype=bool),
                        *np.zeros((2, 0)),
                        true_objs[:, 4],
                    )
                )
            continue

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
        map50 = average_precisions[:, 0].mean()
        map75 = average_precisions[:, 5].mean()
        map50_95 = average_precisions.mean()
    else:
        map50, map75, map50_95 = 0, 0, 0
        average_precisions = []

    return {
        'map50': map50,
        'map75': map75,
        'map50_95': map50_95
    }
    # map50_95, map50, map75, average_precisions)


class DetectionMetric(BaseMetric):
    metric_names: List[str] = ['map50', 'map75', 'map50_95']

    def __init__(self, **kwargs):
        super().__init__()

    def calibrate(self, pred, target, **kwargs):
        result_dict = {k: 0. for k in self.metric_names}

        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        # Gather matching stats for predictions and targets
        for true_objs, predicted_objs in zip(target, pred):
            if predicted_objs.shape[0] == 0:
                if true_objs.shape[0]:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            *np.zeros((2, 0)),
                            true_objs[:, 4],
                        )
                    )
                continue

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
        # map50_95, map50, map75, average_precisions)