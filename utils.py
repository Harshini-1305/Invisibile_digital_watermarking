import cv2
import numpy as np
from PIL import Image

def create_watermark_pattern(image_np, watermark_length, binary_watermark):
    """Create a visual pattern for LSB watermark visualization"""
    pattern_size = 200  # Fixed size for better visibility
    mask = np.zeros((pattern_size, pattern_size), dtype=np.uint8)
    
    # Create a visual pattern from the binary watermark
    block_size = max(1, pattern_size // (watermark_length * 8 // 4))  # Larger blocks for visibility
    idx = 0
    for i in range(0, pattern_size, block_size):
        for j in range(0, pattern_size, block_size):
            if idx < len(binary_watermark):
                # Fill a block with white if bit is 1, black if 0
                if binary_watermark[idx] == '1':
                    mask[i:min(i+block_size, pattern_size), j:min(j+block_size, pattern_size)] = 255
                idx += 1
    
    # Add a border and grid lines for better visibility
    mask[0, :] = 128
    mask[-1, :] = 128
    mask[:, 0] = 128
    mask[:, -1] = 128
    
    # Add grid lines
    for i in range(0, pattern_size, block_size):
        mask[i, :] = 128
        mask[:, i] = 128
        
    # Convert to RGB to add colored text
    mask_rgb = np.stack([mask, mask, mask], axis=2)
    
    # Add explanatory text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(mask_rgb, "LSB Watermark Pattern", (10, 20), font, 0.5, (0, 150, 255), 1, cv2.LINE_AA)
    cv2.putText(mask_rgb, "White = bit '1'", (10, 40), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(mask_rgb, "Black = bit '0'", (10, 60), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    return Image.fromarray(mask_rgb)

def create_svd_pattern(metadata):
    """Create a visualization of SVD watermark regions"""
    pattern_size = 200
    mask = np.zeros((pattern_size, pattern_size, 3), dtype=np.uint8)
    
    region_size = metadata.get('region_size', 128)
    region_map = metadata.get('region_map', [])
    
    # If we have region map, visualize it
    if region_map:
        # Calculate scaling factors
        max_i = max([i for i, _ in region_map]) if region_map else 0
        max_j = max([j for _, j in region_map]) if region_map else 0
        
        scale_h = pattern_size / ((max_i + 1) * region_size) if max_i > 0 else 1
        scale_w = pattern_size / ((max_j + 1) * region_size) if max_j > 0 else 1
        
        # Draw regions
        for i, j in region_map:
            y1 = int(i * region_size * scale_h)
            y2 = int((i + 1) * region_size * scale_h)
            x1 = int(j * region_size * scale_w)
            x2 = int((j + 1) * region_size * scale_w)
            
            # Fill region with green
            mask[y1:y2, x1:x2, 1] = 200  # Green channel
            
            # Add blue border
            mask[y1, x1:x2, 2] = 255  # Blue channel
            mask[y2-1, x1:x2, 2] = 255
            mask[y1:y2, x1, 2] = 255
            mask[y1:y2, x2-1, 2] = 255
    else:
        # If no region map, create a checkerboard pattern
        for i in range(0, pattern_size, 10):
            for j in range(0, pattern_size, 10):
                if (i + j) % 20 < 10:
                    mask[i:i+10, j:j+10, 1] = 200  # Green
    
    # Add explanatory text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(mask, "SVD Watermark Regions", (10, 20), font, 0.5, (0, 150, 255), 1, cv2.LINE_AA)
    cv2.putText(mask, "Green = Modified Regions", (10, 40), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(mask, "Blue = Region Borders", (10, 60), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    return Image.fromarray(mask)

def clean_image(image_np, method):
    """Remove watermark from image based on the watermarking method"""
    cleaned_image_np = image_np.copy()
    
    if method == "LSB (Original)":
        # Clear LSB in all channels
        cleaned_image_np[:, :, 0] = cleaned_image_np[:, :, 0] & 0xFE  # Clear LSB in red channel
        cleaned_image_np[:, :, 1] = cleaned_image_np[:, :, 1] & 0xFE  # Clear LSB in green channel
        cleaned_image_np[:, :, 2] = cleaned_image_np[:, :, 2] & 0xFE  # Clear LSB in blue channel
    else:  # SVD method
        # Apply a slight blur to disrupt the SVD watermark while preserving image quality
        cleaned_image_np = cv2.GaussianBlur(cleaned_image_np, (3, 3), 0.5)
        # Quantize all channels to remove small variations
        cleaned_image_np = (cleaned_image_np // 8) * 8
    
    return Image.fromarray(cleaned_image_np)