import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))

from semseg_eval import SemanticSegmentationEvaluator

from utils import get_used_seqs, unzip_nested


def evaluate_shift_multitask(
    test_annotation_dir, user_submission_dir, max_num_seqs=-1, phase="val"
):
    used_seqs = get_used_seqs(None, split=phase)
    if max_num_seqs > 0 and max_num_seqs < len(used_seqs):
        used_seqs = used_seqs[:max_num_seqs]

    result_dict = {}

    # Semantic segmentation metrics
    if os.path.exists(os.path.join(user_submission_dir, "semseg")):
        print(">> Evaluating semantic segmentation estimation...")
        sem_eval = SemanticSegmentationEvaluator()
        sem_eval.process_from_folder(
            os.path.join(user_submission_dir, "semseg"),
            os.path.join(test_annotation_dir, "semseg"),
            max_num_seqs=max_num_seqs,
            used_seqs=used_seqs,
        )
        sem_result = sem_eval.evaluate()
        result_dict["mIoU"] = sem_result["mIoU"]
        result_dict["mIoU_drop"] = sem_result["mIoU_drop"]
        result_dict["mIoU_source"] = sem_result["start_mIoU"]
        result_dict["mIoU_target"] = sem_result["end_mIoU"]
        print(">> Semantic segmentation results:\n", sem_result)

    return result_dict


def evaluate_shift(test_annotation_dir, user_submission_dir, phase="val"):
    result_dict = {}
    result_dict = evaluate_shift_multitask(
        test_annotation_dir, user_submission_dir, phase=phase
    )
    result_dict["overall"] = result_dict["mIoU"] - 2 * result_dict["mIoU_drop"]
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
    # test
    evaluate(
        "annotations/SHIFT_challenge2023_TTA.zip",
        "testdata/tent.zip",
        "dev",
    )
