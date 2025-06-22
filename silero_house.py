import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from silero_vad import read_audio, get_speech_timestamps, load_silero_vad

# --- 1. OriginalSTFT Module ---
# Reproduces the STFT module where `forward_basis_buffer` is the ONLY
# registered component, and convolutions are performed functionally using
# slices of this buffer as kernels.
model = load_silero_vad(onnx=False)


class OriginalSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window_fn=torch.hann_window):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        if win_length > n_fft:
            raise ValueError("win_length cannot be greater than n_fft")

        # The window function, applied to the basis
        self.window = window_fn(win_length, dtype=torch.float32)

        # Calculate the number of frequency bins for real STFT
        n_freq_bins = self.n_fft // 2 + 1  # For real input, one-sided output

        # --- Generate the `forward_basis_buffer` ---
        # This is exactly how the original model generated its fixed STFT basis.
        # It's a combination of real and imaginary sinusoidal components, shaped
        # as [258, 1, 256] where 258 = 2 * n_freq_bins (129 real + 129 imag).

        # Frequencies for each bin
        freqs = torch.linspace(0, n_fft // 2, n_freq_bins)
        # Time points within the window
        t = torch.arange(win_length).float()

        # Generate real and imaginary components for each frequency bin
        real_basis_filters = torch.zeros(n_freq_bins, 1, win_length)
        imag_basis_filters = torch.zeros(n_freq_bins, 1, win_length)

        for i, freq in enumerate(freqs):
            real_basis_filters[i, 0, :] = (
                torch.cos(2 * np.pi * freq * t / n_fft) * self.window
            )
            imag_basis_filters[i, 0, :] = (
                -torch.sin(2 * np.pi * freq * t / n_fft) * self.window
            )

        # Concatenate real and imaginary parts to form the buffer.
        # This aligns with the [258, 1, 256] shape.
        forward_basis = torch.cat((real_basis_filters, imag_basis_filters), dim=0)

        # Register this tensor as a buffer. It's part of the state_dict but not learnable.
        # This is the ONLY component of STFT that will appear in the state_dict.
        self.register_buffer("forward_basis_buffer", forward_basis)

    def forward(self, x):
        """
        Input: x (Tensor): Audio waveform of shape (batch_size, n_samples)
        Output: Magnitude spectrogram of shape (batch_size, n_freq_bins, n_frames)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension for F.conv1d if mono

        # Apply padding to match `torch.stft`'s default `center=True` behavior.
        pad_amount = self.n_fft // 2
        x_padded = F.pad(x, (pad_amount, pad_amount), mode="reflect")

        # Get the number of frequency bins
        n_freq_bins = self.n_fft // 2 + 1

        # Extract real and imaginary basis from the combined buffer
        real_basis_kernels = self.forward_basis_buffer[:n_freq_bins]
        imag_basis_kernels = self.forward_basis_buffer[n_freq_bins:]

        # Perform the convolutions functionally, using the extracted kernels
        real_spec = F.conv1d(
            x_padded,
            real_basis_kernels,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
        )
        imag_spec = F.conv1d(
            x_padded,
            imag_basis_kernels,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
        )

        # Combine to get magnitude spectrogram
        magnitude_spectrogram = torch.sqrt(real_spec**2 + imag_spec**2)

        return magnitude_spectrogram


# --- 2. SileroVadBlock Module ---
# Reproduces a single block within the encoder Sequential.
# This block contains the 'reparam_conv' and an activation function.
class SileroVadBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        # The Conv1d layer is named 'reparam_conv' within this block
        self.reparam_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        # LeakyReLU with a negative slope of 0.2 is common in Silero models.
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.reparam_conv(x)
        x = self.activation(x)
        return x


# --- 3. VADDecoderRNNJIT Module ---
# Reproduces the VADDecoderRNNJIT module that contains the RNN and final output layers.
class VADDecoderRNNJIT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        # The RNN is named 'rnn' and matches the LSTM parameters.
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # The final output sequential with the 1x1 Conv1d.
        # Identity layers are placeholders to match the `decoder.decoder.2` indexing.
        self.decoder_output_sequential = nn.Sequential(
            nn.Identity(),  # Index 0
            nn.Identity(),  # Index 1
            nn.Conv1d(hidden_size, 1, kernel_size=1, bias=True),  # Index 2
        )

    def forward(self, x, state):
        # RNN expects (batch_size, sequence_length, input_size)
        # Permute input from (batch_size, channels, sequence_length) to (batch_size, sequence_length, channels)
        rnn_input = x.permute(0, 2, 1)

        # Pass input and state to the LSTM
        rnn_output, new_state = self.rnn(rnn_input, state)

        # Output from RNN is (batch_size, sequence_length, hidden_size)
        # Permute back for Conv1d: (batch_size, hidden_size, sequence_length)
        conv_input = rnn_output.permute(0, 2, 1)

        # Apply the final sequential layer
        final_output_conv = self.decoder_output_sequential(
            conv_input
        )  # (batch_size, 1, sequence_length)

        return final_output_conv, new_state


# --- 4. VADRNNJIT Module (The Main Model) ---
# Reproduces the VADRNNJIT model based on the full TorchScript graph.
# This corresponds to the `_model` attribute in the wrapper.
class VADRNNJIT(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=256, hop_length=128, win_length=256):
        super().__init__()

        # Feature extractor (STFT)
        self.stft = OriginalSTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        n_freq_bins = n_fft // 2 + 1

        # Encoder (Sequential of SileroVadBlocks)
        self.encoder = nn.Sequential(
            SileroVadBlock(n_freq_bins, 128, kernel_size=3, padding="same"),
            SileroVadBlock(128, 64, kernel_size=3, padding="same"),
            SileroVadBlock(64, 64, kernel_size=3, padding="same"),
            SileroVadBlock(64, 128, kernel_size=3, padding=0),
        )

        # Decoder (Custom VADDecoderRNNJIT module)
        # LSTM hidden_size determined from state_dict keys (512 / 4 = 128)
        self.decoder = VADDecoderRNNJIT(input_size=128, hidden_size=128, num_layers=1)

    def run_extractors(self, x):
        """
        Corresponds to prim::CallMethod[name="run_extractors"] in the graph.
        This is where the STFT (feature extraction) happens.
        """
        return self.stft(x)

    def forward(self, x, state):
        # 1. Run Extractors (STFT)
        x0 = self.run_extractors(x)  # Output: (batch_size, n_freq_bins, n_frames)

        # 2. Encoder
        x1 = self.encoder(x0)  # Output: (batch_size, 128, n_frames)

        # 3. Decoder
        # x2: output of the decoder, state0: new LSTM hidden/cell states
        x2, state0 = self.decoder(
            x1, state
        )  # x2: (batch_size, 1, n_frames), state0: (h_n, c_n) tuple

        # 4. Squeeze, Mean, Unsqueeze (Post-processing for final score)
        # Squeeze dim 1 to get (batch_size, n_frames)
        squeezed_output = torch.squeeze(x2, 1)

        # Compute mean along the frame dimension (dim=1 here)
        # This calculates an average score per sample in the batch.
        # keepdim=False is default for torch.mean without specifying, matches graph.
        mean_output = torch.mean(squeezed_output, dim=1, keepdim=False)

        # Unsqueeze dim 1 to add back the singleton dimension for the score
        # Output: (batch_size, 1)
        out = torch.unsqueeze(mean_output, 1)

        # The model returns the final output score and the updated recurrent state
        return out, state0


# --- 5. OriginalModelWrapper (Top-Level Model) ---
# This wrapper encapsulates the VADRNNJIT model as an attribute named `_model`.
# This is crucial for matching the `_model.` prefix in the original state_dict keys.
class OriginalModelWrapper(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=256, hop_length=128, win_length=256):
        super().__init__()
        self._model = VADRNNJIT(sample_rate, n_fft, hop_length, win_length)

    def forward(self, x, state):
        return self._model(x, state)


# --- Model Parameters (MUST match original model's training configuration) ---
N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = 256
SAMPLE_RATE = 16000
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = (
    1  # From RNN state_dict (weight_ih/hh sum to 512, implies 128 hidden, 4 for gates)
)

# --- Path to your original state_dict file ---
# IMPORTANT: Replace "path/to/your/original_model.pth" with the actual path.
ORIGINAL_STATE_DICT_PATH = "silero_state_dic.pth"


# --- Main script to test the model ---
def load_model(state_dic):
    print("--- Initializing Model ---")
    model_reproduced = OriginalModelWrapper(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    print("Model initialized successfully.")

    print(f"\n--- Loading state_dict from {state_dic} ---")
    original_state_dict = torch.load(state_dic)
    print("State dict file loaded.")

    # If the state_dict is nested (e.g., {'state_dict': {...}}), extract it
    if "state_dict" in original_state_dict:
        original_state_dict = original_state_dict["state_dict"]
        print("Extracted 'state_dict' from nested dictionary.")

    # --- Key Renaming Logic ---
    # Adjust keys from the loaded state_dict to match the reproduced model's structure.
    new_state_dict = {}
    for k, v in original_state_dict.items():
        base_key = k  # Start with the full key from the loaded state_dict

        # 1. Remove the '_model.' prefix from the outermost wrapper
        # if base_key.startswith("_model."):
        #     base_key = base_key[len("_model.") :]

        # 2. Handle RNN specific renaming: add '_l0' suffix to direct RNN weights/biases
        # The original state_dict has '_model.decoder.rnn.weight_ih' but nn.LSTM uses '_l0'
        # if base_key.startswith("decoder.rnn.") and (
        #     base_key.endswith("weight_ih")
        #     or base_key.endswith("weight_hh")
        #     or base_key.endswith("bias_ih")
        #     or base_key.endswith("bias_hh")
        # ):
        #     new_key = base_key + "_l0"
        # 3. Handle `decoder.decoder.2` mapping to `decoder.decoder_output_sequential.2`
        # This is because the original graph showed `decoder.decoder.2` as a direct path,
        # but in our Python module, it's inside `decoder_output_sequential`.
        if base_key.startswith("_model.decoder.decoder.2."):
            # Replace 'decoder.decoder.2.' with 'decoder.decoder_output_sequential.2.'
            new_key = base_key.replace(
                "decoder.decoder.2.", "decoder.decoder_output_sequential.2."
            )
        else:
            new_key = base_key

        new_state_dict[new_key] = v
    # --- End Key Renaming Logic ---

    print("State dict keys processed for loading.")

    # Load the state dictionary with strict=True for exact matching
    model_reproduced.load_state_dict(new_state_dict, strict=True)
    print("Model state_dict loaded successfully with strict=True!")
    return model_reproduced
