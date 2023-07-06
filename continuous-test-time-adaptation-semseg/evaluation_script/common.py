"""SHIFT base evaluation."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
import tqdm


class Evaluator:
    """Abstract evaluator class."""

    METRICS: list[str] = []

    def __init__(self) -> None:
        """Initialize evaluator."""
        self.reset()

    def reset(self) -> None:
        """Reset evaluator for new round of evaluation."""
        self.metrics = {metric: [] for metric in self.METRICS}

    def process(self, *args: Any) -> None:  # type: ignore
        """Process a batch of data."""
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        """Evaluate all predictions according to given metric.
        Returns:
            dict[str, float]: Evaluation results.
        """
        return {metric: np.nanmean(values) for metric, values in self.metrics.items()}

    def preprocess(self, data: np.array) -> np.array:
        """Preprocess data before evaluation.

        Args:
            data (np.array): Data to be processed.
        Returns:
            np.array: Processed data.
        """
        return data

    def on_next_sequence(self, seq_name: str) -> None:
        """Called when a new sequence is processed.

        Args:
            seq_name (str): Name of the sequence.
        """
        pass

    def append_empty_sample(self, *args: Any, **kwargs: Any) -> None:
        """Process all predictions in a folder of images."""
        raise NotImplementedError

    def process_from_folder(
        self,
        pred_folder_path: str,
        target_folder_path: str,
        max_num_seqs: int = -1,
        used_seqs=None,
    ) -> dict[str, float]:
        """Process all predictions in a folder of images.

        Args:
            pred_folder_path (str): Path to folder containing predictions.
            target_folder_path (str): Path to folder containing targets.

        Returns:
            dict[str, float]: Evaluation results.
        """
        self.reset()
        seqs = sorted(os.listdir(target_folder_path))
        if max_num_seqs > 0:
            seqs = seqs[:max_num_seqs]
        for seq_name in tqdm.tqdm(seqs):
            if used_seqs is not None and seq_name not in used_seqs:
                continue
            self.on_next_sequence(seq_name)
            for frame_name in sorted(
                os.listdir(os.path.join(target_folder_path, seq_name))
            ):
                if not frame_name.endswith(".png"):
                    continue
                try:
                    pred = np.array(
                        Image.open(os.path.join(pred_folder_path, seq_name, frame_name))
                    )
                    target = np.array(
                        Image.open(
                            os.path.join(target_folder_path, seq_name, frame_name)
                        )
                    )
                    frame_id = int(frame_name.split("_")[0])
                    pred = self.preprocess(pred)
                    target = self.preprocess(target)
                    self.process(pred, target, frame_id)

                except Exception as e:
                    print(f"Error when evaluating {seq_name}/{frame_name}: {e}")
                    # apppend empty result
                    self.append_empty_sample(frame_id)
        return self.evaluate()
