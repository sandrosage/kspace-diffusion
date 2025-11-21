from typing import NamedTuple, Optional, Dict, Tuple
import torch
from fastmri.data.subsample import MaskFunc
import numpy as np
from fastmri.data import transforms as T
import fastmri
import torch.nn.functional as F

def normalize_to_minus_one_one(x):
    """
    Normalize a PyTorch tensor to the range [-1, 1].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Tensor normalized to [-1, 1].
    """
    x_min = x.min()
    x_max = x.max()

    # Handle edge case: all values equal
    if x_max == x_min:
        return torch.zeros_like(x)

    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_norm

def pad_center(x: torch.Tensor, target_W: int = 640) -> torch.Tensor:
    """
    Pads the input tensor x along the width dimension (last dimension)
    to the target width with zeros, centering the original content.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [N, C, H, W]
    target_W : int
        Desired width after padding
    
    Returns
    -------
    x_pad : torch.Tensor
        Padded tensor of shape [N, C, H, target_W]
    """
    W = x.shape[-1]
    if target_W == W:
        return x
    
    if target_W < W:
        left  = (W - target_W) // 2
        right = left + target_W
        return x[..., :, left:right] if x.ndim >= 2 and x.shape[-2] == 0 else x[..., left:right]
    
    pad_left  = (target_W - W) // 2
    pad_right = target_W - W - pad_left
    
    x_pad = F.pad(x, (pad_left, pad_right), mode="constant", value=0.0)
    
    return x_pad
    
def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c, c // c * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean

def kspace_to_mri(kspace: torch.Tensor, crop_size: Tuple[int,int] = (320,320)):
    slice_image = fastmri.ifft2c(kspace)
    slice_image_abs = fastmri.complex_abs(slice_image) 
    if crop_size is not None:
        slice_image_abs = fastmri.data.transforms.center_crop(slice_image_abs, crop_size)
    return slice_image_abs

def min_max_normalize(x: torch.Tensor):
    return (x-x.min())/(x.max() - x.min()), torch.stack([x.min(), x.max()], dim=0)

def min_max_unnormalize(x: torch.Tensor, scale: torch.Tensor):
    x_min = scale[0]
    x_max = scale[1]
    return x*(x_max - x_min) + x_min

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

class KspaceUNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """
    full_kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
       
class KspaceUNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, adapt_pool: bool = True, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.adapt_pool = adapt_pool

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> KspaceUNetSample:
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
        if self.adapt_pool:
            kspace_torch = kspace_torch.permute(2, 0, 1).contiguous()
            kspace_torch = pad_center(kspace_torch, target_W=384)
            # kspace_torch = AdaptivePoolTransform((640,384))(kspace_torch)
            kspace_torch = kspace_torch.permute(1, 2, 0).contiguous()

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = KspaceUNetSample(
                full_kspace=kspace_torch,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = KspaceUNetSample(
                full_kspace=kspace_torch,
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample
    
