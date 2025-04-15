from typing import NamedTuple, Optional, Dict, Tuple
import torch
from fastmri.data.subsample import MaskFunc
import numpy as np
from fastmri.data import transforms as T
import fastmri

def reconstruct_kspace(model_output):
    """
    Reconstructs complex k-space from model's log-magnitude and phase output.

    Args:
        model_output (torch.Tensor): Model output (batch, 2, H, W).

    Returns:
        torch.Tensor: Reconstructed k-space (batch, H, W), dtype=torch.complex64.
    """
    log_magnitude = model_output[:, 0, :, :]  # First channel: log-magnitude
    phase = model_output[:, 1, :, :]  # Second channel: phase

    # Convert log-magnitude back to linear scale
    magnitude = torch.exp(log_magnitude)

    # Reconstruct complex k-space using Euler's formula
    kspace_reconstructed = magnitude * torch.exp(1j * phase)

    return kspace_reconstructed  # dtype=torch.complex64

def extract_phase(kspace: torch.Tensor) -> torch.Tensor:
    """
    Extracts the phase from a 2-channel k-space tensor.

    Args:
        kspace (torch.Tensor): Input k-space tensor of shape (batch, 2, H, W),
                               where channel 0 is the real part and channel 1 is the imaginary part.

    Returns:
        torch.Tensor: Phase tensor of shape (batch, 1, H, W), values in range (-π, π).
    """
    real_part = kspace[:, 0, :, :]  # Extract real part
    imag_part = kspace[:, 1, :, :]  # Extract imaginary part

    # Compute phase angle using atan2 (returns values in range [-π, π])
    phase = torch.atan2(imag_part, real_part)

    # Reshape to keep the same format (batch, 1, H, W)
    return phase.unsqueeze(1)  # Add a channel dimension



def preprocess_kspace(kspace: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Converts complex k-space into a two-channel representation: (log-magnitude, phase).

    Args:
        kspace (torch.Tensor): Complex k-space data of shape (batch, height, width).
        epsilon (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Two-channel tensor (batch, 2, height, width).
    """
    log_mag = log_normalize_kspace(kspace, epsilon)
    phase = extract_phase(kspace)

    # Stack into a 2-channel input
    kspace_processed = torch.stack([log_mag, phase], dim=1)  # Shape: (batch, 2, height, width)
    return kspace_processed


def log_normalize_kspace(kspace: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Applies log normalization to the magnitude of k-space data.

    Args:
        kspace (torch.Tensor): Complex k-space data of shape (batch, height, width).
        epsilon (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Log-normalized k-space magnitude.
    """
    magnitude = torch.abs(kspace)  # Compute magnitude
    magnitude = fastmri.complex_abs(kspace)
    log_magnitude = torch.log(magnitude + epsilon)  # Apply log transform

    # Normalize to [0, 1]
    log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())

    return log_magnitude


def complex_center_crop_c_h_w(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def complex_center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger complex image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first complex image.
        y: The second complex image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = complex_center_crop_c_h_w(x, (smallest_height, smallest_width))
    y = complex_center_crop_c_h_w(y, (smallest_height, smallest_width))
    return x, y

class LogPhaseKSample(NamedTuple):
    masked_kspace: torch.Tensor
    kspace: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int,int]
    log_scale: torch.Tensor

class KspaceLDMSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        masked_kspace: Subsampled masked kspace 
        kspace: fully sampled (original) kspace
        target: The target image (if applicable).
    """

    masked_kspace: torch.Tensor
    kspace: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int,int]


class KspaceLDMWithMaskInfoSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        masked_kspace: Subsampled masked kspace 
        kspace: fully sampled (original) kspace
        target: The target image (if applicable).
        acceleration: acceleration factor used by mask_func
        center_fraction: center fraction used by mask_func
    """

    masked_kspace: torch.Tensor
    kspace: torch.Tensor
    target: torch.Tensor
    acceleration: torch.Tensor
    center_fraction: torch.Tensor

class KspaceLDMDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> KspaceLDMSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace = kspace_torch
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

        return KspaceLDMSample(
            masked_kspace=masked_kspace,
            kspace =kspace_torch,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size
        )

class Kspace3DLDMDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> KspaceLDMSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace = kspace_torch
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )
        return KspaceLDMSample(
            masked_kspace=masked_kspace,
            kspace =torch.cat([kspace_torch, torch.zeros(kspace_torch.shape[0], kspace_torch.shape[1], 1)], dim=2),
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size
        )
    
class KspaceLDMDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> KspaceLDMSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace = kspace_torch
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

        return KspaceLDMSample(
            masked_kspace=masked_kspace,
            kspace =kspace_torch,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size
        )
    
class KspaceLDMDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> LogPhaseKSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = T.to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = T.to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        masked_kspace = kspace_torch
        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

        return LogPhaseKSample(
            masked_kspace=masked_kspace,
            kspace =kspace_torch,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            log_scale=
        )

