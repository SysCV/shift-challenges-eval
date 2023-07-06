"""SHIFT depth evaluation."""
from __future__ import annotations

import numpy as np

from common import Evaluator


class SemanticSegmentationEvaluator(Evaluator):
    METRICS = ["mIoU", "mAcc", "start_mIoU", "end_mIoU"]

    def __init__(self, num_classes: int = 23, class_to_ignore: int = 0) -> None:
        """Initialize the semantic segmentation evaluator."""
        self.num_classes = num_classes
        self.class_to_ignore = class_to_ignore
        super().__init__()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self._confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self._confusion_matrix_start = np.zeros((self.num_classes, self.num_classes))
        self._confusion_matrix_end = np.zeros((self.num_classes, self.num_classes))
        self._confusion_matrix_loop_back = np.zeros(
            (self.num_classes, self.num_classes)
        )

    def append_empty_sample(self, frame_id: int) -> None:
        """Append an empty sample to the evaluation."""
        self._confusion_matrix += np.zeros((self.num_classes, self.num_classes))
        if frame_id == 0:
            self._confusion_matrix_start += np.zeros(
                (self.num_classes, self.num_classes)
            )
        elif frame_id == 200:
            self._confusion_matrix_end += np.zeros((self.num_classes, self.num_classes))
        elif frame_id == 400:
            self._confusion_matrix_loop_back += np.zeros(
                (self.num_classes, self.num_classes)
            )

    def calc_confusion_matrix(self, prediction: np.array, target: np.array) -> np.array:
        """Calculate the confusion matrix.
        Args:
            prediction (np.array): Prediction semantic segmentation map, in shape (H, W).
            target (np.array): Target semantic segmentation map, in shape (H, W).
        Returns:
            np.array: Confusion matrix.
        """
        mask = (
            (target >= 0)
            & (target < self.num_classes)
            & (target != self.class_to_ignore)
        )
        return np.bincount(
            self.num_classes * target[mask].astype(np.int32) + prediction[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def preprocess(self, data: np.array) -> np.array:
        if len(data.shape) == 3:
            return data.astype(np.uint8)[:, :, 0]
        return data.astype(np.uint8)

    def process(self, prediction: np.array, target: np.array, frame: int) -> None:
        """Process a batch of data.
        Args:
            prediction (np.array): Prediction semantic segmentation map.
            target (np.array): Target semantic segmentation map.
        """
        self._confusion_matrix += self.calc_confusion_matrix(prediction, target)
        if frame == 0:
            self._confusion_matrix_start += self.calc_confusion_matrix(
                prediction, target
            )
        elif frame == 200:
            self._confusion_matrix_end += self.calc_confusion_matrix(prediction, target)
        elif frame == 400:
            self._confusion_matrix_loop_back += self.calc_confusion_matrix(
                prediction, target
            )

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.
        Returns:
            dict[str, float]: Evaluation results.
        """
        confusion_matrix = self._confusion_matrix
        iou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1)
            + np.sum(confusion_matrix, axis=0)
            - np.diag(confusion_matrix)
        )
        mean_iou = np.nanmean(iou[iou != 0])
        mean_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        confusion_matrix = self._confusion_matrix_start
        iou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1)
            + np.sum(confusion_matrix, axis=0)
            - np.diag(confusion_matrix)
        )
        start_mean_iou = np.nanmean(iou[iou != 0])

        confusion_matrix = self._confusion_matrix_end
        iou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1)
            + np.sum(confusion_matrix, axis=0)
            - np.diag(confusion_matrix)
        )
        end_mean_iou = np.nanmean(iou[iou != 0])

        confusion_matrix = self._confusion_matrix_loop_back
        iou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1)
            + np.sum(confusion_matrix, axis=0)
            - np.diag(confusion_matrix)
        )
        loop_back_mean_iou = np.nanmean(iou[iou != 0])

        return {
            "mIoU": mean_iou * 100,
            "mAcc": mean_acc * 100,
            "start_mIoU": start_mean_iou * 100,
            "end_mIoU": end_mean_iou * 100,
            "loop_back_mIoU": loop_back_mean_iou * 100,
            "mIoU_drop": (start_mean_iou - end_mean_iou) * 100,
        }
