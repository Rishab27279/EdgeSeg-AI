import streamlit as st
import re
import requests
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import os
import gc
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from segment_anything import sam_model_registry, SamPredictor

# Import the comprehensive LLM Query Simplifier
from llm_query_simplifier import QuerySimplifierLLM
from utils import clear_gpu_memory, unload_model_from_memory, get_memory_usage

# Sequential LLM Processing Functions
def load_and_process_llm_query(complex_query: str) -> str:
    """Load LLM, process query, then unload completely"""
    try:
        st.info("🧠 Stage 1: Loading LLM for query simplification...")
        
        # Memory status before loading
        memory_before = get_memory_usage()
        st.info(f"📊 Memory before LLM: {memory_before}")
        
        # Load LLM
        llm_simplifier = QuerySimplifierLLM()
        llm_success = llm_simplifier.download_and_setup()
        
        if not llm_success:
            st.warning("⚠️ LLM loading failed, using rule-based fallback")
            return llm_simplifier.rule_based_simplify(complex_query)
        
        # Memory status after loading
        memory_after = get_memory_usage()
        st.info(f"📊 Memory after LLM load: {memory_after}")
        
        # Process query
        st.info("🔄 Processing complex query with LLM...")
        simplified_prompt = llm_simplifier.simplify_query(complex_query)
        st.success(f"✅ Query simplified: {simplified_prompt}")
        
        # Unload LLM completely
        st.info("🗑️ Unloading LLM to free memory...")
        unload_model_from_memory(llm_simplifier.model)
        unload_model_from_memory(llm_simplifier.tokenizer)
        del llm_simplifier
        
        # Memory status after unloading
        memory_unloaded = get_memory_usage()
        st.success(f"✅ LLM unloaded. Memory: {memory_unloaded}")
        
        return simplified_prompt
        
    except Exception as e:
        st.error(f"❌ LLM processing error: {e}")
        # Fallback to rule-based
        fallback_simplifier = QuerySimplifierLLM()
        return fallback_simplifier.rule_based_simplify(complex_query)

# Configuration class adapted for PC (unchanged)
class Florence2SAMConfig:
    """Configuration class for your fine-tuned model and SAM"""
    def __init__(self):
        # Update these paths for your PC - modify as needed
        self.florence_base_path = "../models/florence2_saved_20250707_204043"  # Adjust path for your PC
        self.florence_adapter_path = os.path.join(self.florence_base_path, "model")
        self.florence_processor_path = os.path.join(self.florence_base_path, "configs")
        
        # SAM model configuration
        self.sam_checkpoint_path = "sam_vit_b_01ec64.pth"
        self.sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Florence2SAMLoader:
    """Load and manage your fine-tuned model and SAM with memory optimization"""
    
    def __init__(self, config: Florence2SAMConfig):
        self.config = config
        self.florence_model = None
        self.florence_processor = None
        self.sam_model = None
        self.sam_predictor = None
        
    def load_florence_model(self):
        """Load your fine-tuned model with LoRA adapter"""
        try:
            st.info("🔄 Loading fine-tuned detection model...")
            
            # Memory status before loading
            memory_before = get_memory_usage()
            st.info(f"📊 Memory before Florence-2: {memory_before}")
            
            # Verify paths exist
            if not os.path.exists(self.config.florence_adapter_path):
                st.error(f"❌ Adapter path not found: {self.config.florence_adapter_path}")
                return False
                
            if not os.path.exists(self.config.florence_processor_path):
                st.error(f"❌ Processor path not found: {self.config.florence_processor_path}")
                return False
            
            # Load base model with flash attention fix and memory optimization
            st.info("📥 Loading base detection model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Fixed: Disable flash attention
                low_cpu_mem_usage=True,  # Memory optimization
                offload_folder="./offload_folder"  # Disk offloading for memory
            )
            st.success("✅ Base model loaded successfully")
            
            # Load LoRA adapter
            st.info("🔧 Loading LoRA adapter...")
            self.florence_model = PeftModel.from_pretrained(base_model, self.config.florence_adapter_path)
            self.florence_model.eval()
            self.florence_model = self.florence_model.half()
            st.success("✅ LoRA adapter loaded successfully")
            
            # Load processor
            st.info("⚙️ Loading processor...")
            self.florence_processor = AutoProcessor.from_pretrained(
                self.config.florence_processor_path, 
                trust_remote_code=True
            )
            st.success("✅ Processor loaded successfully")
            
            # Memory status after loading
            memory_after = get_memory_usage()
            st.info(f"📊 Memory after Florence-2: {memory_after}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading detection model: {e}")
            return False
    
    def load_sam_model(self):
        """Load SAM model for segmentation"""
        try:
            st.info("🔄 Loading segmentation model...")
            
            # Download SAM checkpoint if not exists
            if not os.path.exists(self.config.sam_checkpoint_path):
                st.info("📥 Downloading segmentation checkpoint...")
                response = requests.get(self.config.sam_checkpoint_url)
                with open(self.config.sam_checkpoint_path, 'wb') as f:
                    f.write(response.content)
                st.success("✅ Segmentation checkpoint downloaded")
            
            # Load SAM model
            self.sam_model = sam_model_registry["vit_b"](checkpoint=self.config.sam_checkpoint_path)
            self.sam_predictor = SamPredictor(self.sam_model)
            
            st.success("✅ Segmentation model loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading segmentation model: {e}")
            return False
    
    def load_all_models(self):
        """Load both detection and segmentation models"""
        st.info("🚀 Stage 2: Loading Segmentation models...")
        
        florence_ok = self.load_florence_model()
        sam_ok = self.load_sam_model()
        
        if florence_ok and sam_ok:
            st.success("🎉 All models loaded successfully!")
            return True
        else:
            st.error("❌ Some models failed to load")
            return False
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        st.info("🗑️ Unloading Segmentation models...")
        
        # Unload Florence-2
        if self.florence_model is not None:
            unload_model_from_memory(self.florence_model)
            self.florence_model = None
        
        if self.florence_processor is not None:
            unload_model_from_memory(self.florence_processor)
            self.florence_processor = None
        
        # Unload SAM
        if self.sam_model is not None:
            unload_model_from_memory(self.sam_model)
            self.sam_model = None
        
        if self.sam_predictor is not None:
            unload_model_from_memory(self.sam_predictor)
            self.sam_predictor = None
        
        # Final memory cleanup
        clear_gpu_memory()
        
        memory_final = get_memory_usage()
        st.success(f"✅ All models unloaded. Memory: {memory_final}")

def detect_objects_with_florence(image: np.ndarray, prompt: str, model_loader: Florence2SAMLoader) -> List[Dict]:
    """Use your fine-tuned model for object detection"""
    try:
        if model_loader.florence_model is None or model_loader.florence_processor is None:
            st.warning("⚠️ Detection model not loaded")
            return []
        
        st.info(f"🎯 Using fine-tuned detection model...")
        st.write(f"📝 Input prompt: {prompt}")
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            pil_image = image
        
        # Process inputs
        inputs = model_loader.florence_processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        # Move to device and ensure correct dtype
        device = next(model_loader.florence_model.parameters()).device
        model_dtype = next(model_loader.florence_model.parameters()).dtype
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model_loader.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode output
        generated_text = model_loader.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        st.write(f"🔍 Raw model output: {generated_text}")
        
        # Parse the output to extract detections
        detections = parse_florence_output(generated_text)
        
        st.success(f"✅ Detected {len(detections)} objects")
        return detections
        
    except Exception as e:
        st.error(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return []

def parse_florence_output(output_text: str) -> List[Dict]:
    """Parse output to extract objects and coordinates"""
    try:
        # Remove special tokens
        cleaned = output_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        
        # Extract objects and coordinates using regex
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, cleaned)
        
        detections = []
        for match in matches:
            label = match[0].strip()
            x1, y1, x2, y2 = [int(coord) / 1000.0 for coord in match[1:]]
            detections.append({
                'label': label,
                'bbox_normalized': [x1, y1, x2, y2],
                'confidence': 0.9  # Default confidence score
            })
        
        st.write(f"🔍 Parsed detections: {detections}")
        return detections
        
    except Exception as e:
        st.error(f"❌ Error parsing output: {e}")
        return []

def generate_masks_with_sam(image: np.ndarray, detections: List[Dict], model_loader: Florence2SAMLoader) -> List[np.ndarray]:
    """Use SAM to generate segmentation masks from bounding boxes"""
    try:
        if model_loader.sam_predictor is None:
            st.warning("⚠️ Segmentation model not loaded")
            return []
        
        st.info(f"🎭 Using segmentation model to generate masks...")
        
        # Set image for SAM predictor
        model_loader.sam_predictor.set_image(image)
        
        masks = []
        progress_bar = st.progress(0)
        
        for i, detection in enumerate(detections):
            st.write(f"🔍 Generating mask for {detection['label']}...")
            
            # Convert normalized bbox to pixel coordinates
            h, w = image.shape[:2]
            bbox = detection['bbox_normalized']
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h])]
            
            # SAM expects [x1, y1, x2, y2] format
            box_coords = np.array([x1, y1, x2, y2])
            
            # Generate mask using bounding box prompt
            mask, scores, logits = model_loader.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_coords[None, :],  # Add batch dimension
                multimask_output=False
            )
            
            # Take the first (and only) mask and ensure proper dtype
            mask_processed = mask[0].astype(np.uint8)
            masks.append(mask_processed)
            
            # Update progress
            progress_bar.progress((i + 1) / len(detections))
            
        progress_bar.empty()
        st.success(f"✅ Generated {len(masks)} masks")
        return masks
        
    except Exception as e:
        st.error(f"❌ Mask generation error: {e}")
        import traceback
        traceback.print_exc()
        return []

def sequential_florence_sam_pipeline(image_input, complex_query: str) -> Dict:
    """Sequential pipeline: LLM first → Unload → Florence+SAM → Process → Unload"""
    try:
        st.markdown("## 🔄 Sequential Memory-Optimized Pipeline")
        
        # Create offload folder for memory management
        os.makedirs("./offload_folder", exist_ok=True)
        
        # STAGE 1: LLM Query Processing
        st.markdown("""
        <div class="memory-box">
            <h4>🧠 Stage 1: LLM Query Simplification</h4>
            <p>Loading LLM → Processing query → Unloading LLM</p>
        </div>
        """, unsafe_allow_html=True)
        
        simplified_prompt = load_and_process_llm_query(complex_query)
        
        # Display query conversion
        st.markdown("""
        <div class="llm-box">
            <h4>🔄 Query Conversion Complete</h4>
            <p><strong>Complex Query:</strong> {}</p>
            <p><strong>Simplified Prompt:</strong> {}</p>
        </div>
        """.format(complex_query, simplified_prompt), unsafe_allow_html=True)
        
        # STAGE 2: Segmentation Processing
        st.markdown("""
        <div class="memory-box">
            <h4>🎯 Stage 2: Segmentation Pipeline</h4>
            <p>Loading models → Detection → Segmentation → Unloading models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare image
        if isinstance(image_input, str):
            # URL input
            from utils import load_image_from_url
            image = load_image_from_url(image_input)
            if image is None:
                return {'success': False, 'error': 'Failed to load image from URL'}
        else:
            # File upload
            image = np.array(image_input)
        
        # Load Segmentation models
        config = Florence2SAMConfig()
        model_loader = Florence2SAMLoader(config)
        
        if not model_loader.load_all_models():
            return {'success': False, 'error': 'Failed to load Segmentation models'}
        
        # Stage 2.1: Object detection
        st.markdown("#### 🔄 Stage 2.1: Object Detection")
        detections = detect_objects_with_florence(image, simplified_prompt, model_loader)
        
        if not detections:
            st.warning("⚠️ No objects detected")
            # Unload models before returning
            model_loader.unload_all_models()
            return {
                'success': True,
                'detection_prompt': simplified_prompt,
                'original_image': image,
                'detections': [],
                'masks': [],
                'num_objects': 0
            }
        
        # Stage 2.2: Mask generation
        st.markdown("#### 🔄 Stage 2.2: Mask Generation")
        masks = generate_masks_with_sam(image, detections, model_loader)
        
        # Stage 2.3: Visualization
        st.markdown("#### 🔄 Stage 2.3: Visualization")
        from utils import show_complete_results_streamlit
        show_complete_results_streamlit(image, detections, masks)
        
        # STAGE 3: Model Cleanup
        st.markdown("""
        <div class="memory-box">
            <h4>🗑️ Stage 3: Memory Cleanup</h4>
            <p>Unloading all models to free memory</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Unload all models
        model_loader.unload_all_models()
        
        return {
            'success': True,
            'original_complex_query': complex_query,
            'simplified_prompt': simplified_prompt,
            'detection_prompt': simplified_prompt,
            'original_image': image,
            'detections': detections,
            'masks': masks,
            'num_objects': len(detections),
            'num_masks': len(masks)
        }
        
    except Exception as e:
        st.error(f"❌ Sequential pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
