import streamlit as st
from watermark_methods import lsb_embed_watermark, lsb_extract_watermark, svd_embed_watermark, svd_extract_watermark
from attack_simulations import simulate_attack
from metrics import calculate_psnr, calculate_ssim, calculate_nc
from utils import create_watermark_pattern, create_svd_pattern, clean_image
from PIL import Image
import numpy as np
import tempfile 
import cv2

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