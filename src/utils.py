import streamlit as st
import requests
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import gc
from urllib.parse import urlparse

# Memory Management Functions
def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def unload_model_from_memory(model):
    """Completely unload a model from memory"""
    if model is not None:
        del model
    clear_gpu_memory()

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    else:
        return "CPU mode"

def load_image_from_url(url: str) -> Optional[np.ndarray]:
    """Load actual image from URL using PIL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        st.success(f"✅ Image loaded successfully from URL")
        st.info(f"📸 Image dimensions: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return None

def show_complete_results_streamlit(image: np.ndarray, detections: list, masks: list):
    """Display comprehensive results in Streamlit format"""
    try:
        # Ensure image is in correct format
        if image.max() > 1:
            image_display = image.astype(float) / 255.0
        else:
            image_display = image.astype(float)
        
        # Create visualizations
        st.markdown("## 🎨 Results")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image((image_display * 255).astype(np.uint8), caption="Original Image", use_container_width=True)
        
        with col2:
            st.markdown("### 🌈 Combined Results")
            if masks and detections:
                # Create combined overlay with only segmentation masks (no bounding boxes)
                overlay = image_display.copy()
                colors_rgba = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
                
                for i, mask in enumerate(masks):
                    color = colors_rgba[i % len(colors_rgba)]
                    # Create colored overlay with proper dtype handling
                    mask_float = mask.astype(np.float32)
                    colored_mask = np.zeros_like(overlay, dtype=np.float32)
                    for c in range(3):
                        colored_mask[:, :, c] = mask_float * color[c]
                    
                    # Apply with transparency
                    alpha = 0.4
                    overlay = overlay * (1 - alpha) + colored_mask * alpha
                
                # Create figure with overlay only (no bounding boxes)
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(np.clip(overlay, 0, 1))
                ax.set_title('Segmentation Results', fontsize=16, weight='bold')
                ax.axis('off')
                
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No results to display")
        
        # Print comprehensive summary
        if detections:
            st.markdown("## 📊 Results Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Objects detected", len(detections))
            with col2:
                st.metric("🎭 Masks generated", len(masks))
            
            st.markdown("### 🔍 Detailed Detection Results")
            for i, detection in enumerate(detections, 1):
                with st.expander(f"Object {i}: {detection['label']}"):
                    bbox = detection['bbox_normalized']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Label:** {detection['label']}")
                        st.write(f"**Confidence:** {detection.get('confidence', 0.9):.2f}")
                    
                    with col2:
                        st.write(f"**Bounding Box (normalized):**")
                        st.write(f"- X1: {bbox[0]:.3f}")
                        st.write(f"- Y1: {bbox[1]:.3f}")
                        st.write(f"- X2: {bbox[2]:.3f}")
                        st.write(f"- Y2: {bbox[3]:.3f}")
                    
                    if i <= len(masks):
                        mask_area = masks[i-1].sum()
                        st.write(f"**Mask Area:** {mask_area} pixels")
        
    except Exception as e:
        st.error(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()

def validate_url(url):
    """Validate if URL is properly formatted"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
