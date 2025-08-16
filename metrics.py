import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, attacked):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    if isinstance(original, Image.Image):
        original = np.array(original.convert('RGB'))
    if isinstance(attacked, Image.Image):
        attacked = np.array(attacked.convert('RGB'))
    
    # Ensure images have the same dimensions
    if original.shape != attacked.shape:
        attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))
    
    mse = np.mean((original.astype(np.float64) - attacked.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, attacked):
    """Calculate Structural Similarity Index between two images"""
    if isinstance(original, Image.Image):
        original = np.array(original.convert('RGB'))
    if isinstance(attacked, Image.Image):
        attacked = np.array(attacked.convert('RGB'))
    
    # Ensure images have the same dimensions
    if original.shape != attacked.shape:
        attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for SSIM calculation
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        
    if len(attacked.shape) == 3:
        attacked_gray = cv2.cvtColor(attacked, cv2.COLOR_RGB2GRAY)
    else:
        attacked_gray = attacked
    
    return ssim(original_gray, attacked_gray)

def calculate_nc(original_watermark, extracted_watermark):
    """Calculate Normalized Correlation between original and extracted watermarks"""
    # For text watermarks, convert to binary representation
    if isinstance(original_watermark, str) and isinstance(extracted_watermark, str):
        # Convert strings to binary
        original_binary = ''.join(format(ord(c), '08b') for c in original_watermark)
        extracted_binary = ''.join(format(ord(c), '08b') for c in extracted_watermark)
        
        # Ensure same length by padding the shorter one
        max_len = max(len(original_binary), len(extracted_binary))
        original_binary = original_binary.ljust(max_len, '0')
        extracted_binary = extracted_binary.ljust(max_len, '0')
        
        # Convert to numpy arrays of integers
        original_array = np.array([int(bit) for bit in original_binary])
        extracted_array = np.array([int(bit) for bit in extracted_binary])
    else:
        # If already binary sequences
        original_array = np.array(original_watermark)
        extracted_array = np.array(extracted_watermark)
        
        # Ensure same length
        min_len = min(len(original_array), len(extracted_array))
        original_array = original_array[:min_len]
        extracted_array = extracted_array[:min_len]
    
    # Calculate normalized correlation
    if np.sum(original_array) == 0 or np.sum(extracted_array) == 0:
        return 0.0
    
    correlation = np.sum(original_array * extracted_array) / (
        np.sqrt(np.sum(original_array**2)) * np.sqrt(np.sum(extracted_array**2))
    )
    
    return correlation