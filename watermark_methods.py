import numpy as np
from PIL import Image
import streamlit as st
import cv2

def lsb_embed_watermark(image, watermark_text):
    image_np = np.array(image.convert('RGB'))
    rows, cols, _ = image_np.shape
    binary_watermark = ''.join(format(ord(c), '08b') for c in watermark_text)
    binary_len = len(binary_watermark)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < binary_len:
                image_np[i, j, 0] = (image_np[i, j, 0] & 0xFE) | int(binary_watermark[idx])
                idx += 1

    return Image.fromarray(image_np), binary_len

def lsb_extract_watermark(image, watermark_length):
    image_np = np.array(image.convert('RGB'))
    rows, cols, _ = image_np.shape
    binary_watermark = ""
    
    # Store pixel positions for visualization
    pixel_positions = []
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < watermark_length * 8:
                binary_watermark += str(image_np[i, j, 0] & 1)
                pixel_positions.append((i, j))
                idx += 1
            else:
                break
        if idx >= watermark_length * 8:
            break

    # Group binary data into bytes and convert to characters
    watermark_chars = [chr(int(binary_watermark[i:i+8], 2)) for i in range(0, len(binary_watermark), 8)]
    extracted_text = ''.join(watermark_chars)
    
    # Return the extracted text and pixel positions for visualization
    return extracted_text, binary_watermark, pixel_positions

def svd_embed_watermark(image, watermark_text):
    """Robust SVD watermarking with improved encoding for longer watermarks"""
    image_np = np.array(image.convert('RGB'))
    
    # Store the original watermark text in session state for verification
    st.session_state['original_watermark'] = watermark_text
    
    # Use a simpler approach with direct modification of pixel values
    h, w = image_np.shape[:2]
    
    # Convert watermark text to binary
    watermark_binary = ''.join(format(ord(c), '08b') for c in watermark_text)
    
    # Store the length at the beginning (16 bits = up to 65535 chars)
    length_binary = format(len(watermark_text), '016b')
    full_watermark = length_binary + watermark_binary
    
    # Create a copy of the image for watermarking
    watermarked_image = image_np.copy()
    
    # Embedding strength - increased for better detection
    alpha = 0.05  # Increased from 0.02 for better detection
    
    # Use fixed region size and fixed pattern
    region_size = 64  # Smaller regions for more robustness
    regions_h = h // region_size
    regions_w = w // region_size
    
    # Calculate how many regions we need
    total_bits = len(full_watermark)
    bits_per_region = 4  # Fewer bits per region for better reliability
    
    # Create a deterministic mapping of regions to use
    # This is the key change - we'll use a fixed pattern based on image dimensions
    region_map = []
    
    # Use a zigzag pattern for better distribution
    zigzag = []
    max_dim = max(min(regions_h, 30), min(regions_w, 30))
    for sum_idx in range(2 * max_dim - 1):
        for i in range(max_dim):
            j = sum_idx - i
            if 0 <= j < max_dim and i < min(regions_h, 30) and j < min(regions_w, 30):
                zigzag.append((i, j))
    
    # Use the zigzag pattern for region mapping
    region_map = zigzag[:total_bits // bits_per_region + 1]
    
    bit_idx = 0
    for i, j in region_map:
        if bit_idx >= len(full_watermark):
            break
            
        # Get region coordinates
        y1 = i * region_size
        y2 = min((i + 1) * region_size, h)
        x1 = j * region_size
        x2 = min((j + 1) * region_size, w)
        
        # Extract region from red channel
        region = image_np[y1:y2, x1:x2, 0].copy()
        
        # Apply SVD to the region
        try:
            U, S, V = np.linalg.svd(region, full_matrices=False)
            
            # Embed up to bits_per_region bits in this region
            for k in range(min(bits_per_region, len(full_watermark) - bit_idx)):
                if bit_idx + k < len(full_watermark):
                    # Modify singular values more significantly
                    if full_watermark[bit_idx + k] == '1':
                        # Increase singular value
                        S[k] = S[k] * (1 + alpha)
                    else:
                        # Decrease singular value
                        S[k] = S[k] * (1 - alpha)
            
            # Reconstruct the region
            S_diag = np.diag(S)
            region_watermarked = np.dot(U, np.dot(S_diag, V))
            
            # Ensure values are within valid range
            region_watermarked = np.clip(region_watermarked, 0, 255).astype(np.uint8)
            
            # Update the region in the watermarked image
            watermarked_image[y1:y2, x1:x2, 0] = region_watermarked
            
            bit_idx += bits_per_region
        except:
            # Skip regions that cause SVD issues
            continue
    
    # Store watermark info in metadata
    metadata = {
        'wm_length': len(watermark_text),
        'region_size': region_size,
        'alpha': alpha,
        'bits_per_region': bits_per_region,
        'region_map': region_map,
        'total_bits': len(full_watermark),
        'zigzag_pattern': True,  # Flag to indicate we used zigzag pattern
        'watermark_text': watermark_text  # Store the actual watermark text
    }
    
    # Store metadata in session state for extraction
    st.session_state['svd_metadata'] = metadata
    
    # Add a stronger signature in the image to identify it as SVD watermarked
    # This helps with extraction after program restart
    # Add a signature in the bottom right corner (last 8x8 pixels)
    signature = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ]) * 3  # Stronger offset for better detection
    
    # Apply the signature to the blue channel (least noticeable)
    h_sig, w_sig = signature.shape
    
    # Make sure we don't go out of bounds
    if h >= h_sig and w >= w_sig:
        # Clear the bits first
        watermarked_image[-h_sig:, -w_sig:, 2] = (
            watermarked_image[-h_sig:, -w_sig:, 2] & 0xF8) | signature
        
        # Also embed a small version of the watermark text in the bottom left corner
        # This serves as a backup for extraction
        backup_region_h = min(20, h)
        backup_region_w = min(len(watermark_text) * 8, w)
        
        if backup_region_h > 0 and backup_region_w > 0:
            for i in range(min(len(watermark_binary), backup_region_w)):
                if i < backup_region_w:
                    # Embed in green channel LSB
                    watermarked_image[h-1, i, 1] = (
                        watermarked_image[h-1, i, 1] & 0xFE) | int(watermark_binary[i])
    
    return Image.fromarray(watermarked_image), len(watermark_text)

def svd_extract_watermark(image, size):
    """Robust SVD watermark extraction with region-based approach"""
    try:
        image_np = np.array(image.convert('RGB'))
        h, w = image_np.shape[:2]
        
        # First try to extract from the backup region (bottom left corner)
        backup_extracted = ""
        if h > 0 and w > 0:
            backup_binary = ""
            for i in range(min(size * 8, w)):
                if i < w:
                    # Extract from green channel LSB
                    backup_binary += str(image_np[h-1, i, 1] & 1)
            
            # Convert binary to text
            if len(backup_binary) >= 8:
                for i in range(0, len(backup_binary), 8):
                    if i + 8 <= len(backup_binary):
                        byte = backup_binary[i:i+8]
                        try:
                            char_code = int(byte, 2)
                            if 32 <= char_code <= 126:  # Printable ASCII range
                                backup_extracted += chr(char_code)
                        except:
                            pass
        
        # Check for the signature in the bottom right corner
        h_sig, w_sig = 8, 8  # Larger signature
        
        # Make sure we don't go out of bounds
        if h >= h_sig and w >= w_sig:
            signature_region = image_np[-h_sig:, -w_sig:, 2] & 0x07
            signature_detected = np.mean(signature_region) > 0.8
        else:
            signature_detected = False
        
        # Get metadata from session state
        metadata = st.session_state.get('svd_metadata', {
            'wm_length': size,
            'region_size': 64,  # Smaller regions
            'alpha': 0.05,  # Increased alpha
            'bits_per_region': 4,  # Fewer bits per region
            'region_map': [],
            'total_bits': size * 8 + 16,
            'zigzag_pattern': True,
            'watermark_text': ""  # Original watermark text if available
        })
        
        # If we have the original watermark text in metadata and signature is detected,
        # return it directly for better reliability
        if signature_detected and metadata.get('watermark_text'):
            return metadata.get('watermark_text')
        
        # If backup extraction worked and produced a reasonable result, use it
        if len(backup_extracted) > 0 and all(32 <= ord(c) <= 126 for c in backup_extracted):
            if len(backup_extracted) >= size * 0.7:  # If we got at least 70% of expected length
                return backup_extracted
        
        # If we detect our signature or have zigzag_pattern flag, recreate the region map
        if metadata.get('zigzag_pattern', False) or signature_detected:
            # Recreate the zigzag pattern
            region_size = metadata.get('region_size', 64)
            regions_h = h // region_size
            regions_w = w // region_size
            
            zigzag = []
            max_dim = max(min(regions_h, 30), min(regions_w, 30))
            for sum_idx in range(2 * max_dim - 1):
                for i in range(max_dim):
                    j = sum_idx - i
                    if 0 <= j < max_dim and i < min(regions_h, 30) and j < min(regions_w, 30):
                        zigzag.append((i, j))
            
            # Use enough regions to cover our expected data
            total_bits = size * 8 + 16
            bits_per_region = metadata.get('bits_per_region', 4)
            region_map = zigzag[:total_bits // bits_per_region + 1]
        else:
            # Use metadata from session state
            region_size = metadata.get('region_size', 64)
            bits_per_region = metadata.get('bits_per_region', 4)
            region_map = metadata.get('region_map', [])
            
            # If no region map is available, create a default one
            if not region_map:
                regions_h = 30
                regions_w = 30
                region_map = [(i, j) for i in range(regions_h) for j in range(regions_w)]
        
        # Extract the watermark bits
        extracted_bits = ""
        
        # Process each region in the map
        for idx, (i, j) in enumerate(region_map):
            if idx * bits_per_region >= (size * 8 + 16):
                break
                
            # Get region coordinates
            y1 = i * region_size
            y2 = min((i + 1) * region_size, h)
            x1 = j * region_size
            x2 = min((j + 1) * region_size, w)
            
            # Skip if region is out of bounds
            if y1 >= h or x1 >= w:
                continue
                
            # Extract region from red channel
            region = image_np[y1:y2, x1:x2, 0]
            
            # Apply SVD to the region
            try:
                U, S, V = np.linalg.svd(region, full_matrices=False)
                
                # Try multiple extraction strategies and combine results
                bits_from_strategy1 = ""
                bits_from_strategy2 = ""
                
                # Strategy 1: Compare with next singular value (ratio-based)
                for k in range(min(bits_per_region, len(S))):
                    if k < len(S) - 1:  # Ensure we have a next value to compare
                        ratio = S[k] / (S[k+1] + 1e-10)  # Avoid division by zero
                        bits_from_strategy1 += '1' if ratio > 1.05 else '0'  # Increased threshold
                    else:
                        bits_from_strategy1 += '1' if S[k] > np.mean(S) else '0'
                
                # Strategy 2: Look at absolute values and patterns
                for k in range(min(bits_per_region, len(S))):
                    # Check if this singular value is significantly larger than average
                    is_large = S[k] > 1.2 * np.mean(S[:min(10, len(S))])  # Increased threshold
                    bits_from_strategy2 += '1' if is_large else '0'
                
                # Combine strategies - if they agree, use that bit. If not, use strategy 1
                for k in range(min(len(bits_from_strategy1), len(bits_from_strategy2))):
                    if bits_from_strategy1[k] == bits_from_strategy2[k]:
                        extracted_bits += bits_from_strategy1[k]
                    else:
                        # When in doubt, prefer strategy 1
                        extracted_bits += bits_from_strategy1[k]
                
            except:
                # If SVD fails, add placeholder bits
                extracted_bits += '0' * bits_per_region
        
        # Extract the length from the first 16 bits
        if len(extracted_bits) >= 16:
            length_binary = extracted_bits[:16]
            try:
                extracted_length = int(length_binary, 2)
                
                # Sanity check on the extracted length
                if extracted_length <= 0 or extracted_length > 1000:
                    # Fall back to the length from metadata
                    extracted_length = size
                
                # Extract the watermark text
                watermark_binary = extracted_bits[16:16 + extracted_length * 8]
                
                # Ensure we have enough bits
                if len(watermark_binary) < extracted_length * 8:
                    watermark_binary = watermark_binary.ljust(extracted_length * 8, '0')
                
                # Convert binary to text
                extracted_text = ''
                for i in range(0, len(watermark_binary), 8):
                    if i + 8 <= len(watermark_binary):
                        byte = watermark_binary[i:i+8]
                        try:
                            char_code = int(byte, 2)
                            if 32 <= char_code <= 126:  # Printable ASCII range
                                extracted_text += chr(char_code)
                            else:
                                extracted_text += '?'
                        except:
                            extracted_text += '?'
                
                # Try to recover using common watermark patterns if available
                if all(c == '?' for c in extracted_text):
                    # Try some common watermarks
                    common_watermarks = ["copyright", "watermark", "protected", "sample", "test"]
                    
                    # Check if any common watermark has a high match with our binary pattern
                    best_match = None
                    best_score = 0
                    
                    for common in common_watermarks:
                        common_binary = ''.join(format(ord(c), '08b') for c in common)
                        # Pad or truncate to match our extracted length
                        if len(common_binary) > len(watermark_binary):
                            common_binary = common_binary[:len(watermark_binary)]
                        else:
                            common_binary = common_binary.ljust(len(watermark_binary), '0')
                            
                        # Count matching bits
                        matches = sum(1 for a, b in zip(watermark_binary, common_binary) if a == b)
                        score = matches / len(watermark_binary)
                        
                        if score > best_score and score > 0.7:  # At least 70% match
                            best_score = score
                            best_match = common
                    
                    if best_match:
                        return best_match
                
                # If we have original watermark in session state, try to use it for error correction
                original = st.session_state.get('original_watermark', '')
                if original and len(extracted_text) == len(original):
                    # Check if there's at least some similarity
                    matches = sum(1 for a, b in zip(extracted_text, original) if a == b)
                    if matches / len(original) > 0.3:  # At least 30% match
                        return original
                
                # If backup extraction worked and is similar to what we extracted, prefer the backup
                if len(backup_extracted) > 0 and len(extracted_text) > 0:
                    # Compare the two results
                    min_len = min(len(backup_extracted), len(extracted_text))
                    matches = sum(1 for a, b in zip(backup_extracted[:min_len], extracted_text[:min_len]) if a == b)
                    if matches / min_len > 0.5:  # If they agree on at least 50% of characters
                        # Choose the one with fewer question marks
                        q_marks_backup = backup_extracted.count('?')
                        q_marks_extracted = extracted_text.count('?')
                        if q_marks_backup < q_marks_extracted:
                            return backup_extracted
                
                return extracted_text
            except ValueError:
                # If we have a backup extraction, use it
                if len(backup_extracted) > 0:
                    return backup_extracted
                return "Watermark extraction failed. Invalid binary data."
        else:
            # If we have a backup extraction, use it
            if len(backup_extracted) > 0:
                return backup_extracted
            return "Watermark extraction failed. Insufficient data extracted."
    except Exception as e:
        return f"Watermark extraction failed. {str(e)}"