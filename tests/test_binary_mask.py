from utils import create_binary_mask


def test_create_binary_mask(test_timestamps, wav_file):
    res = create_binary_mask(
        test_timestamps, wav_file, sampling_rate=16000, window_len=31.25e-3
    )
    assert len(res) == 369
