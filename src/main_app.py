import streamlit as st
import json
import numpy as np
from PIL import Image
import io
from urllib.parse import urlparse

from llm_handler import load_and_process_llm_query, sequential_florence_sam_pipeline
from utils import get_memory_usage, validate_url

# Page configuration
st.set_page_config(
    page_title="EdgeSeg AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .llm-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
        font-weight: bold;
    }
    .memory-box {
        background-color: #e8f5e8;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application with sequential model loading"""
    # Header
    st.markdown('<div class="main-header">🧠 EdgeSeg AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;">Democratizing Computer Vision Segmentation with Memory-Optimized Multi-Modal Sequential Processing</div>', unsafe_allow_html=True)
    
    # Memory optimization notice
    st.markdown("""
    <div class="memory-box">
        <h4>🚀 Memory-Optimized Sequential Processing</h4>
        <p>This application uses sequential model loading to minimize RAM usage:</p>
        <ul>
            <li>🧠 Load LLM → Process query → Unload LLM</li>
            <li>🎯 Load Segmentation → Detection & Segmentation → Unload models</li>
            <li>💾 Each model gets full available memory when needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Memory status
        current_memory = get_memory_usage()
        st.info(f"📊 Current Memory: {current_memory}")
        
        # Sequential processing info
        st.markdown("""
        <div class="status-box info-box">
            ✅ Sequential Processing Active<br>
            🧠 Models load/unload automatically<br>
            💾 Memory optimized for low-RAM systems<br>
            🔄 No manual initialization required
        </div>
        """, unsafe_allow_html=True)
        
        st.header("📖 How Sequential Processing Works")
        st.markdown("""
        **Stage 1: Query Simplification**
        1. Load LLM
        2. Process complex query
        3. Unload LLM completely
        
        **Stage 2: Detection & Segmentation**
        1. Load Segmentation models
        2. Run object detection
        3. Generate segmentation masks
        4. Unload all models
        
        **Benefits:**
        - ✅ Works on low-RAM systems
        - ✅ No memory conflicts
        - ✅ Automatic memory management
        """)
        
        st.header("💡 Example Complex Queries")

        # Professional Context Examples
        st.markdown("**Professional Context:**")

        st.markdown("- \"What protective gear do motorcycle riders use?\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/111.png?updatedAt=1752262067954", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/1111.png?updatedAt=1752262082014", caption="Segmented Object",use_container_width=True)

        # Activity-Based Context Examples
        st.markdown("**Activity-Based Context:**")

        st.markdown("- \"Where should I throw the wrapper?\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/555.png?updatedAt=1752266427882", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/5555.png?updatedAt=1752266447020", caption="Segmented Object",use_container_width=True)

        st.markdown("- \"Equipment for displaying information to audiences\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/222.png?updatedAt=1752262579502", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/2222.png?updatedAt=1752262625916", caption="Segmented Object",use_container_width=True)

        # Functional Descriptions Examples
        st.markdown("**Functional Descriptions:**")

        st.markdown("- \"Where should I look to know the speed?\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/aaa.png?updatedAt=1752259527318", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/bbb.png?updatedAt=1752259736090", caption="Segmented Object",use_container_width=True)

        # Conversational Examples
        st.markdown("**Conversational:**")

        st.markdown("- \"I'm looking for something to protect eyes from sun\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/333.png?updatedAt=1752264755341", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/3333.png?updatedAt=1752264769861", caption="Segmented Object",use_container_width=True)

        st.markdown("- \"Help me find what keeps beverages cold in kitchen\"")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ik.imagekit.io/lbiij6kvl/444.png?updatedAt=1752265459579", caption="Original Image",use_container_width=True)
        with col2:
            st.image("https://ik.imagekit.io/lbiij6kvl/4444.png?updatedAt=1752265477679", caption="Segmented Object",use_container_width=True)

    
    # Image input section
    st.markdown('<div class="sub-header">📸 Image Input</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload from Device", "🔗 Load from URL"],
        horizontal=True
    )
    
    uploaded_image = None
    if input_method == "📁 Upload from Device":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image file from your device"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file).convert('RGB')
                st.success(f"✅ Image uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    
    elif input_method == "🔗 Load from URL":
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/image.jpg",
            help="Provide a direct link to an image"
        )
        
        if image_url:
            if validate_url(image_url):
                uploaded_image = image_url  # Store URL for processing
                st.success("✅ Image URL provided")
            else:
                st.error("❌ Please enter a valid URL")
    
    # Complex query input section
    st.markdown('<div class="sub-header">🧠 Advanced Query Input</div>', unsafe_allow_html=True)
    
    complex_query = st.text_area(
        "Enter your complex query in natural language:",
        placeholder="e.g., Show me the safety equipment that construction workers typically wear on their heads",
        help="Describe what you're looking for in natural language - the AI will use sequential processing to handle your request efficiently",
        height=100
    )
    
    # Show image preview
    if uploaded_image is not None and input_method == "📁 Upload from Device":
        st.markdown('<div class="sub-header">📷 Image Preview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Run Sequential Pipeline",
            type="primary",
            use_container_width=True,
            disabled=uploaded_image is None or not complex_query.strip()
        )
    
    # Sequential processing and results
    if process_button and uploaded_image is not None and complex_query.strip():
        st.markdown('<div class="sub-header">🔄 Sequential Memory-Optimized Processing</div>', unsafe_allow_html=True)
        
        # Run sequential pipeline
        results = sequential_florence_sam_pipeline(uploaded_image, complex_query)
        
        if results and results.get('success'):
            # Download options
            if results.get('masks'):
                st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if results['masks']:
                        # Create combined mask for download
                        combined_mask = np.zeros_like(results['masks'][0], dtype=np.uint8)
                        for mask in results['masks']:
                            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                        
                        mask_image = (combined_mask * 255).astype(np.uint8)
                        mask_pil = Image.fromarray(mask_image)
                        buf = io.BytesIO()
                        mask_pil.save(buf, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Combined Mask",
                            data=buf.getvalue(),
                            file_name="combined_segmentation_mask.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                with col2:
                    # Create enhanced results JSON
                    results_json = {
                        'original_complex_query': results.get('original_complex_query', ''),
                        'simplified_prompt': results.get('simplified_prompt', ''),
                        'detection_prompt': results.get('detection_prompt', ''),
                        'num_objects': results['num_objects'],
                        'num_masks': results['num_masks'],
                        'detections': results['detections']
                    }
                    
                    st.download_button(
                        label="📥 Download Enhanced Results",
                        data=json.dumps(results_json, indent=2),
                        file_name="sequential_ai_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
        else:
            st.error("❌ Sequential pipeline failed")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center;">🧠 Powered by Sequential Memory-Optimized Multi-Modal AI Pipeline</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
