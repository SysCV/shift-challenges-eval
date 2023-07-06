import contextlib
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.absolute()))

from scalabel.eval.detect import evaluate_det
from utils import (filer_scalabel_by_frame_id, filter_scalabel, get_used_seqs,
                   load_scalabel, unzip_nested)


def evaluate_shift_multitask(
    test_annotation_dir: str,
    user_submission_dir: str,
    max_num_seqs: int = -1,
    phase: str = "val",
):
    """
    Evaluate SHIFT multitask challenge submission

    Args:
        test_annotation_dir: directory of the test annotation
        user_submission_dir: directory of the user submission
        max_num_seqs: maximum number of sequences to evaluate. If -1, evaluate 
            all sequences
        phase: val or test
    """
    used_seqs = get_used_seqs(split=phase)
    if max_num_seqs > 0 and max_num_seqs < len(used_seqs):
        used_seqs = used_seqs[:max_num_seqs]

    result_dict = {}

    # Object detection
    if os.path.exists(os.path.join(user_submission_dir, "det_2d.json")):
        print(">> Evaluating object detection...")
        det_pred = load_scalabel(
            os.path.join(user_submission_dir, "det_2d.json"), used_seqs
        )
        det_target = load_scalabel(
            os.path.join(test_annotation_dir, "det_2d.json"), used_seqs
        )
        det_pred = filter_scalabel(det_pred, det_target)
        with contextlib.redirect_stdout(io.StringIO()):
            result_all = evaluate_det(
                det_target.frames,
                det_pred.frames,
                det_target.config,
                nproc=1,
            )
        result_all = result_all.summary()
        result_dict["mAP"] = result_all["AP"]

        det_target_start = filer_scalabel_by_frame_id(det_target.copy(), 0, 20)
        det_target_end = filer_scalabel_by_frame_id(det_target.copy(), 180, 220)
        det_target_loop_back = filer_scalabel_by_frame_id(det_target.copy(), 380, 400)
        det_pred_start = filer_scalabel_by_frame_id(det_pred.copy(), 0, 20)
        det_pred_end = filer_scalabel_by_frame_id(det_pred.copy(), 180, 220)
        det_pred_loop_back = filer_scalabel_by_frame_id(det_pred.copy(), 380, 400)
        with contextlib.redirect_stdout(io.StringIO()):
            result_start = evaluate_det(
                det_target_start.frames,
                det_pred_start.frames,
                det_target.config,
                nproc=1,
            )
        result_start = result_start.summary()
        result_dict["mAP_source"] = result_start["AP"]

        with contextlib.redirect_stdout(io.StringIO()):
            result_end = evaluate_det(
                det_target_end.frames,
                det_pred_end.frames,
                det_target.config,
                nproc=1,
            )
        result_end = result_end.summary()
        result_dict["mAP_target"] = result_end["AP"]

        with contextlib.redirect_stdout(io.StringIO()):
            result_loop_back = evaluate_det(
                det_target_loop_back.frames,
                det_pred_loop_back.frames,
                det_target.config,
                nproc=1,
            )
        result_loop_back = result_loop_back.summary()
        result_dict["mAP_loop_back"] = result_loop_back["AP"]
        result_dict["mAP_drop"] = result_dict["mAP_source"] - result_dict["mAP_target"]
        print(">> Object detection results:\n", result_dict)
    return result_dict


def evaluate_shift(test_annotation_dir, user_submission_dir, phase="val"):
    """
    Evaluate SHIFT challenge submission

    Args:
        test_annotation_dir: directory of the test annotation
        user_submission_dir: directory of the user submission
        phase: val or test
    """
    result_dict = {}
    result_dict = evaluate_shift_multitask(
        test_annotation_dir, user_submission_dir, -1, phase=phase
    )
    result_dict["overall"] = result_dict["mAP"] - 2 * result_dict["mAP_drop"]
    return result_dict


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("Starting Evaluation.....")
    print("Evaluation environments:")
    print(" - Python version:", sys.version.split(" ")[0])
    print(" - NumPy version:", np.__version__)

    assert (
        test_annotation_file[-4:] == ".zip"
    ), "Test annotation file should be a zip file"
    assert (
        user_submission_file[-4:] == ".zip"
    ), "User submission file should be a zip file"

    # Unzip the annotation files
    print("Start unzipping...")
    user_submission_dir = user_submission_file[:-4]
    unzip_nested(user_submission_file)
    test_annotation_dir = test_annotation_file[:-4]
    unzip_nested(test_annotation_file)
    print("Unzipping completed.")

    output = {}
    if phase_codename == "dev":
        print("Evaluation phase: Dev")
        result_dict = evaluate_shift(
            test_annotation_dir, user_submission_dir, phase="val"
        )
        output["result"] = [{"val_split": result_dict}]
        output["submission_result"] = output["result"][0]["val_split"]
        print("Completed evaluation for Dev Phase")
        print(result_dict)
    elif phase_codename == "test":
        print("Evaluation phase: Test")
        result_dict = evaluate_shift(
            test_annotation_dir, user_submission_dir, phase="test"
        )
        output["result"] = [{"test_split": result_dict}]
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")
    return output


if __name__ == "__main__":
    evaluate(
        "annotations/SHIFT_challenge2023_TTA.zip",
        "testdata/a7a97c3c.zip",
        "dev",
    )
