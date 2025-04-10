import torch

from fastmri import ifft2c as ifft2c_fastmri
def ifft2c(kspace_data):
    """
    Applies centered inverse 2D FFT to the k-space data.

    Args:
        kspace_data (np.ndarray): Complex-valued k-space data with shape
                                  (num_coils, height, width).

    Returns:
        np.ndarray: Complex-valued image data after IFFT.
    """
    # Perform 2D inverse FFT with FFT shift
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(kspace_data, dim=(-2, -1))), dim=(-2, -1))



def kspace_to_image(kspace_data):
  """
  Converts multi-coil k-space data to image domain.

  Args:
      kspace_data (np.ndarray): Complex-valued k-space data with shape
                                (num_coils, height, width).

  Returns:
      np.ndarray: Image domain data with shape (height, width).
  """
  # Apply inverse FFT to each coil
  coil_images = torch.stack([ifft2c(kspace_data[i]) for i in range(kspace_data.shape[0])], dim=0)

  # Combine coil images
  # combined_image = combine_coil_images(coil_images)

  return coil_images.real
