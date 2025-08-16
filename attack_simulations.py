import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def apply_noise(image_np, intensity=0.5):
    """Apply Gaussian noise to the image"""
    # Convert intensity to standard deviation (0-50)
    stddev = intensity * 50
    noise = np.random.normal(0, stddev, image_np.shape).astype(np.int16)
    noisy_image = np.clip(image_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_blur(image_np, intensity=0.5):
    """Apply Gaussian blur to the image"""
    # Convert intensity to kernel size (must be odd)
    kernel_size = int(intensity * 10) * 2 + 1  # Maps intensity to odd values: 3, 5, 7, 9, 11
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)

def apply_jpeg_compression(image_np, intensity=0.5):
    """Apply JPEG compression to the image"""
    # Use a temporary file with proper cleanup
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.close()  # Close the file before using it with PIL
    
    try:
        # Convert intensity to quality (100 = best, 0 = worst)
        quality = int(100 - (intensity * 90))  # Maps intensity 0→100, 1→10
        
        # Save with determined quality
        temp_img = Image.fromarray(image_np)
        temp_img.save(temp_file.name, format='JPEG', quality=quality)
        
        # Reload the compressed image
        compressed_img = np.array(Image.open(temp_file.name).convert('RGB'))
        
        return compressed_img
    finally:
        # Make sure we close any open file handles before trying to delete
        try:
            # Try to remove the temporary file
            os.unlink(temp_file.name)
        except Exception as e:
            # If deletion fails, just log it but don't crash
            print(f"Warning: Could not delete temporary file {temp_file.name}: {e}")

def apply_crop(image_np, intensity=0.5):
    """Crop a portion of the image and resize back to original dimensions"""
    h, w, _ = image_np.shape
    # Convert intensity to crop percentage (0-30%)
    crop_percent = intensity * 30  
    crop_h = int(h * crop_percent / 100)
    crop_w = int(w * crop_percent / 100)
    
    # Ensure we don't crop too much
    if crop_h * 2 >= h or crop_w * 2 >= w:
        crop_h = min(crop_h, h // 4)
        crop_w = min(crop_w, w // 4)
    
    # Crop from all sides
    cropped = image_np[crop_h:h-crop_h, crop_w:w-crop_w]
    
    # Resize back to original dimensions
    return cv2.resize(cropped, (w, h))

def apply_rotation(image_np, intensity=0.5):
    """Rotate the image by a specified angle"""
    # Convert intensity to angle (0-45 degrees)
    angle = intensity * 45
    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply affine transformation
    rotated = cv2.warpAffine(image_np, rotation_matrix, (w, h), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    return rotated
    
def apply_whatsapp_emulation(image_np):
    """
    Simulates WhatsApp image compression by applying dynamic JPEG quality
    based on image size.
    
    Parameters:
    - image_np: NumPy array of the image
    
    Returns:
    - Compressed image as NumPy array
    """
    # Get image dimensions
    h, w, _ = image_np.shape
    image_size = h * w
    
    # Determine quality based on image size (in megapixels)
    if image_size > 1000000:  # > 1MP (high resolution)
        quality = 35
    elif image_size > 500000:  # > 0.5MP (medium resolution)
        quality = 45
    else:  # Small images
        quality = 55
    
    # Use a temporary file with proper cleanup
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.close()  # Close the file before using it with PIL
    
    try:
        # Save with determined quality
        temp_img = Image.fromarray(image_np)
        temp_img.save(temp_file.name, format='JPEG', quality=quality)
        
        # Reload the compressed image
        compressed_img = np.array(Image.open(temp_file.name).convert('RGB'))
        
        return compressed_img
    finally:
        # Make sure we close any open file handles before trying to delete
        try:
            # Try to remove the temporary file
            os.unlink(temp_file.name)
        except Exception as e:
            # If deletion fails, just log it but don't crash
            print(f"Warning: Could not delete temporary file {temp_file.name}: {e}")

def simulate_attack(image, attack_type, intensity=0.5):
    """
    Simulates various attacks on watermarked images to test robustness.
    
    Parameters:
    - image: PIL Image object
    - attack_type: String indicating the type of attack
    - intensity: Float between 0 and 1 indicating attack strength
    
    Returns:
    - Attacked image as PIL Image
    """
    image_np = np.array(image.convert('RGB'))
    
    if attack_type == "Noise":
        image_np = apply_noise(image_np, intensity)
    elif attack_type == "Blur":
        image_np = apply_blur(image_np, intensity)
    elif attack_type == "JPEG Compression":
        image_np = apply_jpeg_compression(image_np, intensity)
    elif attack_type == "Crop":
        image_np = apply_crop(image_np, intensity)
    elif attack_type == "Rotation":
        image_np = apply_rotation(image_np, intensity)
    elif attack_type == "WhatsApp Emulation":
        image_np = apply_whatsapp_emulation(image_np)
    
    return Image.fromarray(image_np)