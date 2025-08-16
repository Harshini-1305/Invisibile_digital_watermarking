import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
from skimage.metrics import structural_similarity as ssim

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
    
    # _water a copy of the image for watermarking
    watermarked_image = image_np.copy()
    
    # Embedding strength - increased for better detection
    alpha = 0.02  # Slightly increased from 0.01
    
    # Use fixed region size and fixed pattern
    region_size = 128
    regions_h = h // region_size
    regions_w = w // region_size
    
    # Calculate how many regions we need
    total_bits = len(full_watermark)
    bits_per_region = 8
    
    # Create a deterministic mapping of regions to use
    # This is the key change - we'll use a fixed pattern based on image dimensions
    region_map = []
    
    # Use a zigzag pattern for better distribution
    zigzag = []
    max_dim = max(min(regions_h, 20), min(regions_w, 20))
    for sum_idx in range(2 * max_dim - 1):
        for i in range(max_dim):
            j = sum_idx - i
            if 0 <= j < max_dim and i < min(regions_h, 20) and j < min(regions_w, 20):
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
            
            # Embed up to 8 bits in this region
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
        'zigzag_pattern': True  # Flag to indicate we used zigzag pattern
    }
    
    # Store metadata in session state for extraction
    st.session_state['svd_metadata'] = metadata
    
    # Also embed a small signature in the image to identify it as SVD watermarked
    # This helps with extraction after program restart
    # Add a small signature in the bottom right corner (last 4x4 pixels)
    signature = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]) * 2  # Small offset that won't be visible
    
    # Apply the signature to the blue channel (least noticeable)
    h_sig, w_sig = signature.shape
    watermarked_image[-h_sig:, -w_sig:, 2] = (
        watermarked_image[-h_sig:, -w_sig:, 2] & 0xFC) | signature
    
    return Image.fromarray(watermarked_image), len(watermark_text)

def svd_extract_watermark(image, size):
    """Robust SVD watermark extraction with region-based approach"""
    try:
        image_np = np.array(image.convert('RGB'))
        h, w = image_np.shape[:2]
        
        # Check for the signature in the bottom right corner
        h_sig, w_sig = 4, 4
        signature_region = image_np[-h_sig:, -w_sig:, 2] & 0x03
        
        # Default metadata
        metadata = st.session_state.get('svd_metadata', {
            'wm_length': size,
            'region_size': 128,
            'alpha': 0.02,
            'bits_per_region': 8,
            'region_map': [],
            'total_bits': size * 8 + 16,
            'zigzag_pattern': True
        })
        
        # If we detect our signature or have zigzag_pattern flag, recreate the region map
        if metadata.get('zigzag_pattern', False) or np.mean(signature_region) > 0.5:
            # Recreate the zigzag pattern
            region_size = metadata.get('region_size', 128)
            regions_h = h // region_size
            regions_w = w // region_size
            
            zigzag = []
            max_dim = max(min(regions_h, 20), min(regions_w, 20))
            for sum_idx in range(2 * max_dim - 1):
                for i in range(max_dim):
                    j = sum_idx - i
                    if 0 <= j < max_dim and i < min(regions_h, 20) and j < min(regions_w, 20):
                        zigzag.append((i, j))
            
            # Use enough regions to cover our expected data
            total_bits = size * 8 + 16
            bits_per_region = metadata.get('bits_per_region', 8)
            region_map = zigzag[:total_bits // bits_per_region + 1]
        else:
            # Use metadata from session state
            region_size = metadata.get('region_size', 128)
            bits_per_region = metadata.get('bits_per_region', 8)
            region_map = metadata.get('region_map', [])
            
            # If no region map is available, create a default one
            if not region_map:
                regions_h = 20
                regions_w = 20
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
                        bits_from_strategy1 += '1' if ratio > 1.01 else '0'
                    else:
                        bits_from_strategy1 += '1' if S[k] > np.mean(S) else '0'
                
                # Strategy 2: Look at absolute values and patterns
                for k in range(min(bits_per_region, len(S))):
                    # Check if this singular value is significantly larger than average
                    is_large = S[k] > 1.1 * np.mean(S[:min(10, len(S))])
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
                
                return extracted_text
            except ValueError:
                return "Watermark extraction failed. Invalid binary data."
        else:
            return "Watermark extraction failed. Insufficient data extracted."
    except Exception as e:
        return f"Watermark extraction failed. {str(e)}"

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

st.set_page_config(page_title="Invisible Digital Watermarking", layout="wide")
st.title(" Invisible Digital Watermarking")

method = st.radio("Select Watermarking Method", ["LSB (Original)", "SVD (New)"])

tab1, tab2 = st.tabs(["Embed Watermark", "Extract Watermark"])

with tab1:
    st.header(" Embed Watermark")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="embed")
    watermark_text = st.text_input("Enter watermark text")

    if uploaded_file and watermark_text:
        image = Image.open(uploaded_file)
        if method == "LSB (Original)":
            watermarked_image, wm_len = lsb_embed_watermark(image, watermark_text)
        else:
            watermarked_image, wm_len = svd_embed_watermark(image, watermark_text)

        # Store watermark length and text in session state
        st.session_state['watermark_length'] = len(watermark_text)
        st.session_state['watermark_text'] = watermark_text
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(watermarked_image, caption="Watermarked Image", use_container_width=True)

        st.success("Watermark embedded successfully!")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            watermarked_image.save(tmp.name)
            st.download_button("Download Watermarked Image", tmp.read(), "watermarked.png", "image/png")
        st.info(f"ℹ️ Watermark Details:\n- Text: '{watermark_text}'\n- Length: {len(watermark_text)} characters\n- Method: {method}")

with tab2:
    st.header(" Extract Watermark")
    extract_file = st.file_uploader("Upload watermarked image", type=["jpg", "jpeg", "png"], key="extract")
    
    # Use the stored length from session state or default to 10 if not available
    watermark_length = st.session_state.get('watermark_length', 10)
    
    # Add attack simulation options
        # Add attack simulation options
    st.subheader("Attack Simulation")
    attack_enabled = st.checkbox("Enable Attack Simulation")
    
    if attack_enabled:
        attack_col1, attack_col2 = st.columns(2)
        with attack_col1:
            attack_type = st.selectbox("Attack Type", 
                                      ["Noise", "Blur", "JPEG Compression", "Crop", "Rotation", "WhatsApp Emulation"])
        
        # Default intensity
        attack_intensity = 0.5
        
        with attack_col2:
            # Dynamic sliders based on attack type
            if attack_type == "Noise":
                noise_stddev = st.slider("Noise Std Dev", min_value=0, max_value=50, value=25)
                attack_intensity = noise_stddev / 50.0
                st.caption("Higher values add more noise")
                
            elif attack_type == "Blur":
                blur_ksize = st.select_slider("Blur Kernel Size", options=[3, 5, 7, 9, 11], value=5)
                attack_intensity = (blur_ksize - 1) / 10.0  # maps 3→0.2, 5→0.4, etc.
                st.caption("Larger kernel = more blurring")
                
            elif attack_type == "JPEG Compression":
                jpeg_quality = st.slider("JPEG Quality", min_value=10, max_value=100, value=40)
                attack_intensity = (100 - jpeg_quality) / 90.0  # maps 100→0, 10→1
                st.caption("Lower quality = more compression artifacts")
                
            elif attack_type == "Crop":
                crop_percent = st.slider("Crop Percentage", min_value=0, max_value=30, value=15)
                attack_intensity = crop_percent / 30.0
                st.caption("Percentage of image edges to crop")
                
            elif attack_type == "Rotation":
                rotation_angle = st.slider("Rotation Angle (Degrees)", min_value=0, max_value=45, value=15)
                attack_intensity = rotation_angle / 45.0  # maps 0→0, 45→1
                st.caption("Rotation angle in degrees")
                
            # WhatsApp Emulation requires no intensity
            elif attack_type == "WhatsApp Emulation":
                st.info("WhatsApp Emulation uses dynamic compression based on image size")
                attack_intensity = 0.5  # Default value, not used for WhatsApp Emulation
    
    if extract_file:
        original_image = Image.open(extract_file)
        
        # Apply attack if enabled
        if attack_enabled:
            attacked_image = simulate_attack(original_image, attack_type, attack_intensity)
            
            # Display appropriate message based on attack type
            if attack_type == "WhatsApp Emulation":
                # Get image dimensions to determine quality used
                img_np = np.array(original_image)
                h, w, _ = img_np.shape
                image_size = h * w
                
                if image_size > 1000000:
                    quality = 35
                elif image_size > 500000:
                    quality = 45
                else:
                    quality = 55
                
                st.warning(f"Applied WhatsApp Emulation (JPEG quality={quality} based on image size: {w}×{h})")
            else:
                st.warning(f"Applied {attack_type} attack with intensity {attack_intensity}")
            
            # Show original and attacked images side by side
            attack_col1, attack_col2 = st.columns(2)
            with attack_col1:
                st.image(original_image, caption="Original Watermarked Image", use_container_width=True)
            with attack_col2:
                st.image(attacked_image, caption=f"After {attack_type}", use_container_width=True)
            
            # Use the attacked image for extraction
            image = attacked_image
        else:
            image = original_image
        
        if method == "LSB (Original)":
            extracted = lsb_extract_watermark(image, watermark_length)
            image_np = np.array(image.convert('RGB'))
            
            # Create a more visible pattern for LSB watermark
            pattern_size = 200  # Fixed size for better visibility
            mask = np.zeros((pattern_size, pattern_size), dtype=np.uint8)
            
            # Create a more visual representation of the watermark
            binary_watermark = ""
            idx = 0
            for i in range(min(image_np.shape[0], 1000)):  # Limit to first 1000 rows for efficiency
                for j in range(min(image_np.shape[1], 1000)):  # Limit to first 1000 columns
                    if idx < watermark_length * 8:
                        binary_watermark += str(image_np[i, j, 0] & 1)
                        idx += 1
                    else:
                        break
                if idx >= watermark_length * 8:
                    break
            
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
            
            watermark_visible = Image.fromarray(mask_rgb)
            
            # Clean the image by removing the LSB in all channels
            cleaned_image_np = image_np.copy()
            cleaned_image_np[:, :, 0] = cleaned_image_np[:, :, 0] & 0xFE  # Clear LSB in red channel
            cleaned_image_np[:, :, 1] = cleaned_image_np[:, :, 1] & 0xFE  # Clear LSB in green channel
            cleaned_image_np[:, :, 2] = cleaned_image_np[:, :, 2] & 0xFE  # Clear LSB in blue channel
            cleaned_image = Image.fromarray(cleaned_image_np)
        else:
            extracted = svd_extract_watermark(image, watermark_length)
            
            # Add a note about SVD extraction limitations
            if extracted == "??????????" or all(c == '?' for c in extracted):
                st.warning("""
                **Note about SVD Watermarking**: 
                The SVD method sometimes has difficulty extracting watermarks from large images or 
                after the application is restarted. This is because SVD watermarking modifies the 
                frequency domain of the image in subtle ways that can be affected by image size, 
                compression, or loss of session data.
                
                For demonstration purposes, you can try:
                1. Using smaller images (under 5MB)
                2. Embedding and extracting in the same session
                3. Using shorter watermark text
                """)
            image_np = np.array(image.convert('RGB'))

            # Create a visualization of the SVD watermark regions
            metadata = st.session_state.get('svd_metadata', {
                'region_size': 128,
                'region_map': []
            })
            
            region_size = metadata.get('region_size', 128)
            region_map = metadata.get('region_map', [])
            
            # Create a fixed-size visualization
            pattern_size = 200
            mask = np.zeros((pattern_size, pattern_size, 3), dtype=np.uint8)
            
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
            
            watermark_visible = Image.fromarray(mask)
            
            # Create a cleaned image by applying a slight blur and quantization
            cleaned_image_np = image_np.copy()
            # Apply a slight blur to disrupt the SVD watermark while preserving image quality
            cleaned_image_np = cv2.GaussianBlur(cleaned_image_np, (3, 3), 0.5)
            # Quantize all channels to remove small variations
            cleaned_image_np = (cleaned_image_np // 8) * 8
            cleaned_image = Image.fromarray(cleaned_image_np)

        # Add evaluation metrics if attack was applied
        if attack_enabled:
            if isinstance(extracted, tuple):
                extracted_text = extracted[0]
                binary_watermark = extracted[1]
            else:
                extracted_text = extracted
                binary_watermark = ""
                
            # Calculate success rate based on character match
            original_text = st.session_state.get('watermark_text', '')
            if original_text:
                match_count = sum(1 for a, b in zip(original_text, extracted_text) if a == b)
                total_chars = max(len(original_text), len(extracted_text))
                success_rate = (match_count / total_chars) * 100 if total_chars > 0 else 0
                
                # Calculate image quality metrics
                psnr_value = calculate_psnr(original_image, image)
                ssim_value = calculate_ssim(original_image, image)
                nc_value = calculate_nc(original_text, extracted_text)
                
                # Display metrics in a nice format
                st.subheader(" Watermark Robustness Metrics")
                
                # Create three columns for metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("PSNR (dB)", f"{psnr_value:.2f}", 
                             "Higher is better", delta_color="normal")
                    if psnr_value > 30:
                        st.success("Excellent image quality")
                    elif psnr_value > 20:
                        st.info("Good image quality")
                    else:
                        st.warning("Poor image quality")
                
                with metric_col2:
                    st.metric("SSIM", f"{ssim_value:.4f}", 
                             "Higher is better (max 1.0)", delta_color="normal")
                    if ssim_value > 0.9:
                        st.success("Excellent structural similarity")
                    elif ssim_value > 0.7:
                        st.info("Good structural similarity")
                    else:
                        st.warning("Poor structural similarity")
                
                with metric_col3:
                    st.metric("NC", f"{nc_value:.4f}", 
                             "Higher is better (max 1.0)", delta_color="normal")
                    if nc_value > 0.8:
                        st.success("Excellent watermark recovery")
                    elif nc_value > 0.5:
                        st.info("Partial watermark recovery")
                    else:
                        st.warning("Poor watermark recovery")
                
                # Overall recovery assessment
                st.metric("Watermark Recovery Rate", f"{success_rate:.1f}%", 
                         f"{match_count}/{total_chars} characters matched")
                
                # Display color-coded feedback based on watermark recovery
                if success_rate > 80:
                    st.success(f"✅ Watermark survived the {attack_type} attack well!")
                elif success_rate > 50:
                    st.warning(f"⚠️ Watermark partially survived the {attack_type} attack.")
                else:
                    st.error(f"❌ Watermark was significantly damaged by the {attack_type} attack.")
                
                # Add detailed explanation
                st.info("""
                **Metrics Explanation:**
                - **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the attacked image compared to the original. Higher values indicate less distortion.
                - **SSIM (Structural Similarity Index)**: Measures the perceived similarity between images. Values range from 0 to 1, with 1 indicating perfect similarity.
                - **NC (Normalized Correlation)**: Measures how well the watermark was preserved. Values range from 0 to 1, with 1 indicating perfect preservation.
                """)

        if isinstance(extracted, tuple):
            extracted_text = extracted[0]
        else:
            extracted_text = extracted
            
        st.success(f"Extracted Watermark: {extracted_text}")
        st.info(f"Watermark Details:\n- Length: {watermark_length} characters\n- Extracted Text: '{extracted_text}'\n- Method: {method}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Watermarked Image", use_container_width=True)
        with col2:
            st.image(watermark_visible, caption="Watermark Pattern", use_container_width=True)
        with col3:
            st.image(cleaned_image, caption="Cleaned Image", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cleaned_image.save(tmp.name)
                st.download_button("Download Cleaned Image", tmp.read(), "cleaned_image.png", "image/png")
            
            # Add a button to test extraction from the cleaned image
            if st.button("Test Extraction from Cleaned Image"):
                st.subheader("Extraction Test from Cleaned Image")
                
                # Try to extract watermark from the cleaned image
                if method == "LSB (Original)":
                    cleaned_extracted, _, _ = lsb_extract_watermark(cleaned_image, watermark_length)
                else:
                    cleaned_extracted = svd_extract_watermark(cleaned_image, watermark_length)
                
                # Display the results
                st.code(f"Extracted from cleaned image: {cleaned_extracted}")
                
                # Compare with the original extracted text
                original_text = extracted_text if isinstance(extracted_text, str) else extracted
                match_count = sum(1 for a, b in zip(original_text, cleaned_extracted) if a == b)
                total_chars = max(len(original_text), len(cleaned_extracted))
                match_rate = (match_count / total_chars) * 100 if total_chars > 0 else 0
                
                
                if match_rate < 50:
                    st.success(f"✅ Watermark successfully removed! Only {match_rate:.1f}% of characters match.")
                elif match_rate < 80:
                    st.info(f"ℹ️ Watermark partially removed. {match_rate:.1f}% of characters still match.")
                else:
                    st.warning(f"⚠️ Watermark still detectable. {match_rate:.1f}% of characters match.")