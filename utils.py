import re
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import csv
from typing import Union, Dict
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


# compute model size
def compute_model_size_and_params(model):
    if isinstance(model, tuple):
        return model
    num_params = np.sum([param.numel() for param in model.parameters()])
    dtype = next(model.parameters()).dtype
    num_bits = int(re.search(r"\d+$", str(dtype)).group(0))
    size_in_bytes = float(num_params * num_bits / (8 * 1e6))
    return size_in_bytes, int(num_params)


def timeit(func):
    def internal(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        lag = end - start
        return res, lag

    return internal


# compute metric for this sample


def compute_metrics(y_true, y_pred, latency):
    min_len = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:min_len])
    y_pred = np.array(y_pred[:min_len])
    prec, recall, f1 = (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )
    res = {"prec": prec, "recall": recall, "f1": f1, "latency": latency}
    return res


def dump_results(file_path, summary: list):
    lines_to_dump = []
    if not os.path.exists(file_path):
        header = [
            "time",
            "model",
            "prec",
            "recall",
            "f1",
            "test_dataset",
            "num_params",
            "size(MB)",
            "latency",
        ]
        lines_to_dump.append(header)
    lines_to_dump.append(summary)
    with open(file_path, "a+") as f:
        writer = csv.writer(f)
        for line in lines_to_dump:
            writer.writerow(line)


def create_binary_mask(timestamps: Union[Dict[str, float]], window_len=31.5e-3):
    """
    timestamps  should be an iterable of format [{"start": float, "end":float} ... ]
    """
    timestamps_mask = []

    # Determine the maximum end time across both labels and valid_preds
    # This ensures the mask covers the full extent of relevant activity.
    max_end_time = 0.0
    max_end_time = max(max_end_time, max(d["end"] for d in timestamps))

    # If there are no segments at all, return empty masks
    if max_end_time == 0.0:
        return [], []

    cursor = 0.0

    # The `+ step * 0.5` add buffer for float innacuracy
    while cursor < max_end_time + window_len * 0.5:  # Run slightly beyond max_end_time
        # Check if the current cursor position falls within any ground truth segment
        is_speech = 0
        for segment in timestamps:
            # Check if cursor is within [start, end)
            if segment["start"] <= cursor < segment["end"]:
                is_speech = 1
                break  # Found an overlapping label, no need to check further for this cursor
        timestamps_mask.append(is_speech)
        cursor += window_len  # Move to the next time step

    return timestamps_mask


def parse_wavs_labels(path_dir: str | Path, label_ext="scv"):
    wav_label_dic = {
        "wav": sorted(
            [
                os.path.join(path_dir, f)
                for f in os.listdir(path_dir)
                if f.endswith(".wav")
            ],
            key=lambda f: (
                int(re.search(r"(\d+)", f).group(1))
                if re.search(r"(\d+)", f)
                else float("inf")
            ),
        ),
        "labels": sorted(
            [
                os.path.join(path_dir, f)
                for f in os.listdir(path_dir)
                if f.endswith(f"{label_ext}")
            ],
            key=lambda f: (
                int(re.search(r"(\d+)", f).group(1))
                if re.search(r"(\d+)", f)
                else float("inf")
            ),
        ),
    }
    return wav_label_dic


def time_latency(func):
    def internal(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        lag = end - start
        return res, lag

    return internal


def aggregate_results(res, metadata):
    prec = np.mean([dic["prec"] for dic in res])
    recall = np.mean([dic["recall"] for dic in res])
    f1 = np.mean([dic["f1"] for dic in res])
    latency = np.mean([dic["latency"] for dic in res])
    dataset = metadata["dataset_name"]
    model_name = metadata["model_name"]
    model_size = metadata["model_size"]
    num_params = metadata["num_params"]
    date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    return date, model_name, dataset, prec, recall, f1, num_params, model_size, latency


@contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    yield
    os.chdir(old_dir)
