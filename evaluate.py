from abc import ABC, abstractmethod
from utils import (
    parse_wavs_labels,
    time_latency,
    create_binary_mask,
    compute_metrics,
    compute_model_size_and_params,
    dump_results,
    aggregate_results,
)
from tr_vad.infer_model import VADInferrer
import sys
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from subprocess import run
import glob
import os
import re
from pathlib import Path
from pyannote.audio import Pipeline

from silero_vad import  get_speech_timestamps,  read_audio, load_silero_vad

get_speech_timestamps = time_latency(
    get_speech_timestamps
)  # decorate to compute latency
TEN_VAD_PATH = "./ten-vad/examples/"
TEN_VAD_DATASET_PATH = "ten-vad/testset"
TR_VAD_CHKP_PATH = (
    "/home/yehoshua/projects/silero-vad/tr_vad/checkpoint/weights_10_acc_97.09.pth"
)
RES_PATH = "results.csv"

BLOB_DURATION = 31.25e-3
BLOB_SAMPLES = 500

silero_model = load_silero_vad(onnx=False)
pyannote_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=os.environ.get("HF_TOKEN"),
)
pyannote_model = pyannote_pipeline._segmentation.model

pyannote_pipeline = time_latency(pyannote_pipeline)  # decorate to compute latency


class BaseOutputProcessor(ABC):

    @abstractmethod
    def process_output(output_file):
        pass


class RTTMProcessor(BaseOutputProcessor):

    @classmethod
    def process_output(cls, output_path):
        df = pd.read_csv(
            output_path,
            sep=" ",
            names=[
                "type",
                "file_id",
                "channel_id",
                "start",
                "duration",
                "orthography",
                "speaker_type",
                "speaker_name",
                "confidence_score",
                "other",
            ],
        )
        df["start"] = df["start"].astype(float)
        df["duration"] = df["duration"].astype(float)
        df["end"] = df["start"] + df["duration"]
        return df

    @classmethod
    def extract_timestamps(cls, df):
        return [
            {"start": float(df["start"][i]), "end": float(df["end"][i])}
            for i in range(len(df["start"]))
        ]


class VADDataset(ABC):
    def __init__(self, window_len=BLOB_DURATION):
        self.labels = self.process_labels(window_len=window_len)

    def load_dataset(self, dir_path: str, **kwargs):
        self.dataset = parse_wavs_labels(dir_path, **kwargs)
        return self.dataset

    @abstractmethod
    def process_labels(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class TenVadDataset(VADDataset):

    def __init__(self, dataset_path, window_len=BLOB_DURATION):
        self.dataset_path = dataset_path
        self.files_dic = parse_wavs_labels(dataset_path)
        super().__init__(window_len=window_len)

    def process_labels(self, window_len=BLOB_DURATION):
        labels = []
        for label_path in self.files_dic["labels"]:
            with open(label_path, "r") as f:
                label = f.read().split(",")[1:]
                label = [
                    {"start": float(s), "end": float(e), "voice": float(v)}
                    for s, e, v in [
                        [label[i], label[i + 1], label[i + 2]]
                        for i in range(0, len(label) - 2, 3)
                    ]
                ]
                filtered_label = [
                    {k: float(v) for k, v in lab.items() if lab["voice"] == 1}
                    for lab in label
                ]
                filtered_label = [dic for dic in filtered_label if dic != dict()]
                label = create_binary_mask(filtered_label, window_len=window_len)
                labels.append(label)
        return labels

    def __len__(self):
        return len(self.files_dic["wav"])

    def __getitem__(self, i):
        return {
            "wav": self.files_dic.get("wav")[i],
            "labels": self.labels[i],
        }

    def __repr__(self):
        return "Ten_Vad_Dataset"


class MSDWILDDataset(VADDataset):
    def __init__(
        self, dataset_path, dataset_partition="many.val.rttm", window_len=BLOB_DURATION
    ):
        self.dataset_path = dataset_path
        self.dataset_partition = dataset_partition
        self.files_dic = self.create_mapping()
        super().__init__(window_len=window_len)

    def create_mapping(self):
        rttm_path = os.path.join(self.dataset_path, "rttms", self.dataset_partition)
        rttm_df = RTTMProcessor.process_output(rttm_path)
        rttm_index = rttm_df["file_id"].unique().tolist()
        wavs = []
        labels = []
        print("loading dataset")
        for idx in tqdm(rttm_index):
            filtered_df = (
                rttm_df[rttm_df["file_id"] == idx].reset_index().sort_values(by="start")
            )
            timestamps = RTTMProcessor.extract_timestamps(filtered_df)
            wav = glob.glob(os.path.join(self.dataset_path, "wav", f"*{idx}.wav"))[0]
            wavs.append(wav)
            labels.append(timestamps)
        files_dic = {"wav": wavs, "labels": labels}
        return files_dic

    def process_labels(self, window_len=BLOB_DURATION):
        labels = []
        for label in self.files_dic["labels"]:
            label = create_binary_mask(label, window_len=window_len)
            labels.append(label)
        return labels

    def __repr__(self):
        return "MSDWILDDataset"

    def __len__(self):
        return len(self.files_dic["wav"])

    def __getitem__(self, i):
        return {"wav": self.files_dic["wav"][i], "label": self.labels[i]}


class ModelRunner(ABC):
    _model = None

    def __init__(self, dataset: VADDataset):
        self.dataset = dataset
        self.preds = list()
        self.labels = self.dataset.labels
        self.metadata = dict()
        if self._model is None:
            raise ValueError(
                "in you class definition you should define a static attribute _model"
            )

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def binarize_outputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def run_benchmark(self, output_file=None):
        res = []
        exp_metadata = {}
        assert self.dataset is not None, ValueError(
            "you should load the dataset before, run load_dir"
        )
        assert self.labels is not None, ValueError(
            "You should load labels before, run  load_labels()"
        )
        labels = self.dataset.labels
        model_size, num_params = compute_model_size_and_params(self._model)
        exp_metadata["model_size"] = model_size
        exp_metadata["num_params"] = num_params
        exp_metadata["model_size"] = model_size
        exp_metadata["num_params"] = num_params
        exp_metadata["dataset_name"] = self.dataset.__repr__()
        exp_metadata["model_name"] = self.__repr__()
        preds, latencies = self.run_inference()
        res = [
            compute_metrics(label, pred, latency)
            for label, pred, latency in zip(labels, preds, latencies)
        ]
        summary = aggregate_results(res, metadata=exp_metadata)
        if output_file:
            dump_results(output_file, summary)
        return res, exp_metadata


class SileroRunner(ModelRunner):

    _model = silero_model

    def __init__(self, dataset: VADDataset):
        super().__init__(dataset)

    def load_wav(self, wav_path):
        wav = read_audio(wav_path)
        return wav

    def forward(self, wav, sampling_rate=16000, return_seconds=True):
        metadata = {}
        speech_timestamps_and_latency = get_speech_timestamps(
            wav,
            silero_model,
            sampling_rate=sampling_rate,
            return_seconds=return_seconds,
        )
        if isinstance(speech_timestamps_and_latency, tuple):
            speech_timestamps, latency = speech_timestamps_and_latency
            metadata["latency"] = latency
        else:
            assert isinstance(speech_timestamps_and_latency, list)
            speech_timestamps = speech_timestamps_and_latency
        return speech_timestamps, metadata

    def run_inference(self):
        preds = []
        latencies = []

        for sample in tqdm(self.dataset):
            wav = read_audio(sample["wav"])
            speech_timestamps, metadata = self.forward(wav)
            if latency := metadata.get("latency", None):
                latencies.append(latency)
            pred = self.binarize_outputs(speech_timestamps)
            preds.append(pred)
        self.preds = preds
        self.metadata["latencies"] = latencies
        return preds, latencies

    def binarize_outputs(self, timestamps, window_len=BLOB_DURATION):
        return create_binary_mask(timestamps, window_len=window_len)

    def __repr__(self):
        return "Silero-V5"


class TenVadRunner(ModelRunner):
    _model = (0.0350, None)

    def __init__(self, dataset: VADDataset):
        super().__init__(dataset)

    def forward(
        self, wav_path, path_of_module=TEN_VAD_PATH, window_sample=BLOB_SAMPLES
    ):
        """
        check the script https://github.com/yecohn/ten-vad/blob/main/examples/test.py,
        you should clone the repo and run the script from the repo (change dir in subprocess)
        it will create an output_file with extension ten_vad_pred.csv in the folder of the wavs file
        step: this is the number of sample / window ~ 31.24 ms by default (see silero benchmark: https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
        """
        path = Path(wav_path)
        full_path = str(path.absolute())
        result = run(
            [sys.executable, "test.py", full_path, str(window_sample)],
            cwd=path_of_module,  # Explicitly tell subprocess where to run from
            check=True,
            capture_output=True,  # Capture output for better debugging
            text=True,  # Decode output as text
        )
        print(result.stdout)

    def binarize_outputs(self, output_file):
        df = pd.read_csv(output_file)
        preds = df.iloc[:, 1].tolist()
        latency = np.sum(df.iloc[:, 2].tolist())
        return preds, latency

    def load_preds(self, preds_dir):
        preds_files = sorted(
            glob.glob(os.path.join(preds_dir, "*_ten_vad_pred.csv")),
            key=lambda x: int(re.search("\d+", x).group(0)),
        )
        preds, latencies = zip(
            *(self.binarize_outputs(pred_file) for pred_file in preds_files)
        )
        self.preds = preds
        self.metadata["latencies"] = latencies
        return preds, latencies

    def run_inference(self):

        for sample in tqdm(self.dataset):
            wav_file = sample["wav"]
            self.forward(wav_file)
        preds, latencies = self.load_preds(Path(wav_file).parent)
        return preds, latencies

    def __repr__(self):
        return "Ten_Vad"


class PyannoteRunner(ModelRunner):
    _model = pyannote_model

    def __init__(self, dataset: VADDataset):
        super().__init__(dataset)
        self._forward_processor = RTTMProcessor

    def forward(self, wav_path):
        tmp_path = "tmp.csv"
        # TODO: if remove decorator time_latency need to remove latency
        pyannote_output, latency = pyannote_pipeline(wav_path)
        with open(tmp_path, "w") as f:
            pyannote_output.write_rttm(f)
        df = self._forward_processor.process_output(tmp_path)
        timestamps = self._forward_processor.extract_timestamps(df)
        os.remove(tmp_path)
        return timestamps, latency

    def run_inference(self, window_len=BLOB_DURATION):
        preds = []
        latencies = []
        for sample in tqdm(self.dataset):
            wav_file = sample["wav"]
            timestamps, latency = self.forward(wav_file)
            pred = self.binarize_outputs(timestamps, window_len=window_len)
            preds.append(pred)
            latencies.append(latency)
        self.preds = preds
        self.metadata["latencies"] = latencies
        return preds, latencies

    def binarize_outputs(self, timestamps: torch.Tensor, window_len=BLOB_DURATION):
        return create_binary_mask(timestamps, window_len=window_len)

    def __repr__(self):
        return "Pyannote"


class TRVADRunner(ModelRunner):
    _model = None

    def __init__(self, dataset: VADDataset, checkpoint_path, quantize=False):
        self.inferer = VADInferrer(checkpoint_path=checkpoint_path, quantize=quantize)
        self.checkpoint_path = checkpoint_path
        self._model = self.inferer.model
        self.quantize = quantize
        super().__init__(dataset)

    def forward(self, wav_path, window_sample=BLOB_SAMPLES):

        wav_path = str(Path(wav_path).absolute())
        pred, latency = self.inferer.infer_vad(
            audio_path=wav_path, len_blob=window_sample
        )
        return pred, latency

    def load_preds(self, output_file):
        df = pd.read_csv(output_file)
        latency = df["lag"]
        pred = [int(elem) for elem in df.iloc[0, 2:]]
        return pred, latency

    def run_inference(self, window_sample=BLOB_SAMPLES):
        preds = []
        latencies = []
        for sample in tqdm(self.dataset):
            wav_file = sample["wav"]
            pred, latency = self.forward(wav_file, window_sample=window_sample)
            if pred == 0:
                continue
            preds.append(pred)
            latencies.append(latency)
        self.preds = preds
        self.metadata["latencies"] = latencies
        return preds, latencies

    def binarize_outputs(self):
        return NotImplemented

    def __repr__(self):
        if self.quantize:
            return "TR_VAD_int8"
        else:
            return "TR_VAD"


if __name__ == "__main__":
    # dataset = MSDWILDDataset(
    #     dataset_path="/home/yehoshua/.cache/huggingface/datasets/MSDWILD"
    # )

    dataset = TenVadDataset(dataset_path=TEN_VAD_DATASET_PATH)
    silero_runner = SileroRunner(dataset=dataset)
    silero_runner.run_benchmark(output_file=RES_PATH)
    # ten_vad_runner = TenVadRunner(dataset=dataset)
    # ten_vad_runner.run_benchmark(output_file=RES_PATH)
    # pyannote_runner = PyannoteRunner(dataset=dataset)
    # pyannote_runner.run_benchmark(output_file=RES_PATH)
    # tr_vad_runner = TRVADRunner(
    #     dataset=dataset, checkpoint_path=TR_VAD_CHKP_PATH, quantize=True
    # )
    # tr_vad_runner.run_benchmark(output_file=RES_PATH)
