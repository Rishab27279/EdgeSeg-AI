EdgeSeg AI - Memory-Efficient Prompt-Based Segmentation
<p align="center"> <strong>A lightweight, memory-optimized approach to complex prompt-based image segmentation that delivers accurate results on resource-constrained devices.</strong> </p>
🚀 Overview
EdgeSeg AI tackles the challenge of performing sophisticated prompt-based image segmentation with limited computational resources. Unlike traditional approaches that require multiple models running simultaneously, our solution uses a sequential model loading strategy to minimize memory usage while maintaining segmentation accuracy.

🎯 Key Features
Memory-Efficient Architecture - Sequential model loading reduces RAM usage by 60-70%

Tricky Prompt Handling - Converts complex natural language queries into actionable segmentation tasks

Interactive Prompt Refinement - Users can review and modify simplified prompts before segmentation

High-Quality Segmentation - Combines Fine-Tuned Florence-2 with SAM for precise object boundaries

Resource-Friendly - Optimized for free-tier users and standard consumer hardware

🏗️ Architecture
Sequential Processing Pipeline
text
Complex Prompt → LLM Simplification → Florence-2 Detection → SAM Segmentation
     ↓              ↓                    ↓                   ↓
  Load LLM    →  Unload LLM      →   Load Florence-2   →   Generate Masks
                                      Load SAM Base
Model Components
Component	Purpose	Memory Impact
LLM	Complex prompt simplification	Loaded only during prompt processing
Fine-Tuned Florence-2	Bounding box prediction from simplified prompts	Loaded during detection phase
SAM Base	Precise mask generation using bounding boxes	Loaded during segmentation phase
🔧 How It Works
1. Prompt Simplification
Input: Complex natural language prompt (e.g., "What protective gear do motorcycle riders use?")

Process: LLM converts complex queries into clear, actionable descriptions

Output: Simplified prompt optimized for object detection

User Control: Review and modify simplified prompts before proceeding

2. Object Detection
Unload: LLM is removed from memory

Load: Fine-Tuned Florence-2 model for bounding box prediction

Process: Generate precise bounding boxes from simplified prompts

Advantage: Better accuracy through prompt optimization

3. Segmentation
Load: SAM Base model alongside Florence-2

Process: Use bounding boxes as guidance for mask generation

Output: High-quality segmentation masks

Efficiency: Targeted segmentation reduces computational overhead

💡 Innovation
Memory Optimization Strategy
Traditional approaches load all models simultaneously:

text
VLM + DINO + SAM = High Memory Usage (Problematic for free-tier users)
Our approach uses sequential loading:

text
LLM → Unload → Florence-2 + SAM = Reduced Memory Usage (Accessible to all users)
Performance vs. Efficiency Trade-off
Sacrifices: Slight reduction in processing speed due to model loading/unloading

Gains: 60-70% reduction in peak memory usage

Result: Democratized access to advanced segmentation capabilities

🎨 Example Results
Safety Equipment Query
Original Prompt: "What protective gear do motorcycle riders use?"

Simplified Prompt: "motorcycle helmet and protective gear"

Original Image	Segmented Result
![Original](https://s://ik.imagekit.io/rompt**: "Where should I throw the wrapper?"	
Simplified Prompt: "trash bin or waste container"

Original Image	Segmented Result
![Original](https://ik.imagekit.io/l Usage	
Requirements
Python 3.8+

CUDA-compatible GPU (optional but recommended)

8GB+ RAM

Installation
bash
git clone https://github.com/your-username/EdgeSeg-AI.git
cd EdgeSeg-AI
pip install -r requirements.txt
Usage
python
from edgeseg import EdgeSegAI

# Initialize the segmentation pipeline
segmenter = EdgeSegAI()

# Process complex prompt
result = segmenter.segment_image(
    image_path="path/to/image.jpg",
    prompt="What protective gear do motorcycle riders use?"
)

# Review simplified prompt (optional)
print(f"Simplified prompt: {result.simplified_prompt}")

# Get segmentation mask
mask = result.segmentation_mask
📊 Performance Metrics
Metric	Traditional Approach	EdgeSeg AI	Improvement
Peak Memory Usage	12-16GB	4-6GB	60-70% reduction
Minimum GPU Memory	8GB	4GB	50% reduction
Processing Time	2-3 seconds	4-6 seconds	Trade-off for efficiency
Segmentation Quality	High	High	Maintained
🙏 Acknowledgments
This project is inspired by the groundbreaking work in the LLM-Seg paper by the research team who pioneered the concept of using large language models for image segmentation tasks. Their novel architecture and innovative approach laid the foundation for this work.

While the original LLM-Seg implementation achieves superior accuracy through simultaneous model deployment, our adaptation prioritizes computational efficiency to make advanced segmentation accessible to users with limited resources.

Hats off to the LLM-Seg authors for their exceptional contribution to the field and for inspiring this resource-optimized implementation.

🤝 Contributing
We welcome contributions! Whether it's:

🐛 Bug fixes

✨ New features

📖 Documentation improvements

🎨 UI/UX enhancements

Please feel free to open issues and submit pull requests.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

⭐ Support
If you find this project helpful:

Give it a star ⭐ - It would mean the world to me!

Share your feedback - Comments and suggestions are always welcome

Try it out - Your usage and feedback help improve the project

Your support motivates continued development and helps make advanced AI accessible to everyone.

<p align="center"> <strong>Made with ❤️ for the community</strong><br> <em>Bringing advanced AI capabilities to resource-constrained environments</em> </p>
