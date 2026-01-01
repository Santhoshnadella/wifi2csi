import numpy as np
from scipy.signal import butter, lfilter
import torch

def butterfly_filter(data, cutoff=0.1, fs=100, order=5):
    """
    Low-pass filter to remove high frequency noise from CSI streams.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data, axis=-1)
    return y

def preprocess_csi_frame(csi_data):
    """
    Takes raw CSI data and prepares it for the model.
    Input: [Antennas, Subcarriers, 2, TimeSteps] (numpy)
    Output: Tensor [1, Antennas, Subcarriers, 2, TimeSteps]
    """
    # 1. Denoise (Optional, simple low-pass on time axis)
    # Assumes last dim is time
    # csi_clean = butterfly_filter(csi_data, cutoff=20, fs=100)
    csi_clean = csi_data # Skip for synthetic/fast demo

    # 2. Normalize
    # Simple min-max or z-score. Here we just scale to reasonable range if needed.
    # Transformer inputs often benefit from standardized inputs.
    mean = np.mean(csi_clean)
    std = np.std(csi_clean) + 1e-6
    csi_norm = (csi_clean - mean) / std

    # 3. Convert to Tensor
    tensor = torch.from_numpy(csi_norm).float()
    
    # 4. Add Batch Dimension if missing
    # Expected model input: [Batch, Antennas, Subcarriers, 2, TimeSteps]
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    
    return tensor

def extract_amplitude_phase(csi_complex):
    """
    Convert complex (Real, Imag) to (Amp, Phase)
    csi_complex shape: [..., 2, ...] where 2 is (real, imag)
    """
    # This is for visualization/analysis, not necessarily for the model
    # The model provided takes raw Re/Im as input (size 2 on dim 3 for the reshaping).
    
    real = csi_complex[..., 0, :]
    imag = csi_complex[..., 1, :]
    
    amp = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)
    
    return amp, phase
