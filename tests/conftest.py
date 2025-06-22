from pytest import fixture
import librosa
import json


with open("tests/testset-audio-01_timestamps..json", "r") as f:
    timestamps = json.load(f)


@fixture
def wav_file():
    return "./ten-vad/testset/testset-audio-01.wav"


@fixture
def wav():
    return librosa.load("./ten-vad/testset/testset-audio-01.wav")


@fixture
def test_timestamps():
    return timestamps
