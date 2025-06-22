import re
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import csv
from typing import Union, Dict
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import librosa

FILE_NAME = os.environ.get("FILE_NAME")


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
    with open(f"{FILE_NAME}_outputs_preds.txt", "a") as f1:
        f1.write(f"{y_true}\n{y_pred}")
        f1.write("\n\n")

    prec, recall, f1 = (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    if os.path.exists("f{FILE_NAME}_confusion.csv"):
        os.remove("f{FILE_NAME}_confusion.csv")
    with open(f"{FILE_NAME}_confusion.csv", "a") as f:
        f.write(f"{tn}, {fp}, {fn}, {tp}\n")
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


def create_binary_mask(
    timestamps: Union[Dict[str, float]], wav_file, sampling_rate, window_len=31.5e-3
):
    """
    timestamps  should be an iterable of format [{"start": float, "end":float} ... ]
    """
    wav, _ = librosa.load(wav_file, sr=sampling_rate)
    duration = len(wav) / sampling_rate
    number_bins = int((duration // window_len) + 1)
    mask = np.zeros(number_bins)

    # The `+ step * 0.5` add buffer for float innacuracy
    for i in range(len(mask)):
        is_speech = 0
        for (
            segment
        ) in timestamps:  # we need this loop in the case we have multiple speakers
            # Check if cursor is within [start, end)
            if segment["start"] <= i * window_len < segment["end"]:
                is_speech = 1
                break  # Found an overlapping label, no need to check further for this cursor
        mask[i] = is_speech
    return mask


def parse_wavs_labels(path_dir: str | Path, label_ext="scv", stop=None):
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
