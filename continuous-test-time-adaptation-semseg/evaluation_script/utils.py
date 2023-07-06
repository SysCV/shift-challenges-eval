import copy
import csv
import os
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))

from scalabel.label.io import load


CONDITIONS = [
    "clear_to_rainy",
    "clear_to_foggy",
    "daytime_to_night",
]
SEQ_INFO_PATH_VAL = os.path.join(
    str(Path(__file__).parent.absolute()), "val_front_images_seq.csv"
)
SEQ_INFO_PATH_TEST = os.path.join(
    str(Path(__file__).parent.absolute()), "test_front_images_seq.csv"
)


# Load sequence info
with open(SEQ_INFO_PATH_VAL, "r") as f:
    reader = csv.DictReader(f)
    SEQ_INFO_VAL = [
        (
            row["video"],
            row["shift_type"],
            row["start_weather_coarse"],
            row["start_timeofday_coarse"],
        )
        for row in reader
    ]
with open(SEQ_INFO_PATH_TEST, "r") as f:
    reader = csv.DictReader(f)
    SEQ_INFO_TEST = [
        (
            row["video"],
            row["shift_type"],
            row["start_weather_coarse"],
            row["start_timeofday_coarse"],
        )
        for row in reader
    ]


def get_used_seqs(seq_filter, split="val"):
    """Get used sequences for evaluation."""
    if split == "val":
        SEQ_INFO = SEQ_INFO_VAL
    elif split == "test":
        SEQ_INFO = SEQ_INFO_TEST
    if seq_filter is not None:
        used_seqs = []
        for seq in SEQ_INFO:
            if seq_filter == seq[1] and seq[2] == "clear" and seq[3] == "daytime":
                used_seqs.append(seq[0])
    else:
        used_seqs = []
        for seq in SEQ_INFO:
            if seq[2] == "clear" and seq[3] == "daytime":
                used_seqs.append(seq[0])
    return used_seqs


def unzip_nested(file_path):
    """Unzip nested zip files."""
    assert file_path.endswith(".zip"), "Not a zip file"

    output_path = file_path[:-4]
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    for file in os.listdir(output_path):
        if file.endswith(".zip"):
            unzip_nested(os.path.join(output_path, file))


SCALABEL_CACHE = {}


def load_scalabel(file_path, used_seqs=None):
    """Load scalabel."""
    if file_path in SCALABEL_CACHE:
        data_ = copy.deepcopy(SCALABEL_CACHE[file_path])
    else:
        data = load(file_path, validate_frames=False)
        SCALABEL_CACHE[file_path] = data
        data_ = copy.deepcopy(data)
    if used_seqs is not None:
        data_.frames = [frame for frame in data_.frames if frame.videoName in used_seqs]
    return data_


def filter_scalabel(pred, target):
    """Filter the scalabel by target."""
    used_frames = []
    for frame in target.frames:
        used_frames.append(frame.videoName + frame.name)
    used_frames = set(used_frames)
    pred.frames = [
        frame for frame in pred.frames if frame.videoName + frame.name in used_frames
    ]
    return pred


def filer_scalabel_by_frame_id(pred, frame_id_start, frame_id_end):
    """Filter the scalabel by frame id."""
    used_frames = []
    for frame in pred.frames:
        if frame.frameIndex >= frame_id_start and frame.frameIndex <= frame_id_end:
            used_frames.append(frame.videoName + frame.name)
    used_frames = set(used_frames)
    pred.frames = [
        frame for frame in pred.frames if frame.videoName + frame.name in used_frames
    ]
    return pred
