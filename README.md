# EdgeSeg-AI : Memory-Efficient Prompt-Based Segmentation

<p align="center">
  <strong>A lightweight, memory-optimized approach to complex prompt-based image segmentation that delivers accurate results on resource-constrained devices.</strong>
</p>

---

## 🚀 Overview

EdgeSeg-AI tackles the challenge of performing sophisticated prompt-based image segmentation with limited computational resources. Unlike traditional approaches that require multiple models running simultaneously, our solution uses a **sequential model loading strategy** to minimize memory usage while maintaining segmentation accuracy.

## 🎯 Key Features

- **Memory-Efficient Architecture**: Sequential model loading reduces RAM usage by 60-70%.
- **Tricky Prompt Handling**: Converts complex natural language queries into actionable segmentation tasks.
- **Interactive Prompt Refinement**: Users can review and modify simplified prompts before segmentation.
- **High-Quality Segmentation**: Combines Fine-Tuned Florence-2 with SAM for precise object boundaries.
- **Resource-Friendly**: Optimized for free-tier users and standard consumer hardware.

## 🏗️ Architecture

### Sequential Processing Pipeline
  `Complex Prompt → LLM Simplification → Florence-2 Detection → SAM Segmentation
   ↓ ↓ ↓ ↓
   Load LLM → Unload LLM → Load Florence-2 → Generate Masks
   Load SAM Base`

### Model Components

| Component                | Purpose                                        | Memory Impact                        |
| ------------------------ | ---------------------------------------------- | ------------------------------------ |
| **LLM**                  | Complex prompt simplification                  | Loaded only during prompt processing |
| **Fine-Tuned Florence-2**| Bounding box prediction from prompts           | Loaded during detection phase        |
| **SAM Base**             | Precise mask generation using bounding boxes    | Loaded during segmentation phase     |

## 🔧 How It Works

### 1. Prompt Simplification

- **Input**: Complex natural language prompt (e.g., "What protective gear do motorcycle riders use?")
- **Process**: LLM converts complex queries into clear, actionable descriptions.
- **Output**: Simplified prompt optimized for object detection.
- **User Control**: Review and modify simplified prompts before proceeding.

### 2. Object Detection

- **Unload**: LLM is removed from memory.
- **Load**: Fine-Tuned Florence-2 model for bounding box prediction.
- **Process**: Generate precise bounding boxes from simplified prompts.
- **Advantage**: Better accuracy through prompt optimization.

### 3. Segmentation

- **Load**: SAM Base model alongside Florence-2.
- **Process**: Use bounding boxes as guidance for mask generation.
- **Output**: High-quality segmentation masks.
- **Efficiency**: Targeted segmentation reduces computational overhead.

## 💡 Innovation

### Memory Optimization Strategy

**Traditional approaches** load all models simultaneously:
     `VLM + DINO + SAM = High Memory Usage (Problematic for free-tier users)`

**Our approach** uses sequential loading:
     `LLM → Unload → Florence-2 + SAM = Reduced Memory Usage (Accessible to all users)`

### Performance vs. Efficiency Trade-off

- **Sacrifices**: Slight reduction in processing speed due to model loading/unloading.
- **Gains**: 60-70% reduction in peak memory usage.
- **Result**: Democratized access to advanced segmentation capabilities.

## 🎨 Example Results

### Safety Equipment Query

- **Original Prompt**: "Where should I look to know the speed?"
- **Simplified Prompt**: "Speedometer in car/bike/other-vehicles"

| Original Image | Segmented Result |
| :------------: | :--------------: |
| ![Motorcycle Rider](https://ik.imagekit.io/lbiij6kvl/aaa.png?updatedAt=1752259527318) | ![Segmented Gear](https://ik.imagekit.io/lbiij6kvl/bbb.png?updatedAt=1752259736090) |

### Waste Context

- **Original Prompt**: "Where should I throw the wrapper?"
- **Simplified Prompt**: "trash bin or waste container"

| Original Image | Segmented Result |
| :------------: | :--------------: |
| ![Wrapper](https://ik.imagekit.io/lbiij6kvl/555.png?updatedAt=1752266427882) | ![Trash Bin](https://ik.imagekit.io/lbiij6kvl/5555.png?updatedAt=1752266447020) |


## 🛠️ Installation & Usage

### Requirements

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM

### Installation
     git clone https://github.com/your-username/EdgeSeg-AI.git
     cd EdgeSeg-AI
     pip install -r requirements.txt

### Usage
     from edgeseg import EdgeSegAI
     
     Initialize the segmentation pipeline
     segmenter = EdgeSegAI()
     
     Process complex prompt
     result = segmenter.segment_image(
     image_path="path/to/image.jpg",
     prompt="What protective gear do motorcycle riders use?"
     )
     
     Review simplified prompt (optional)
     print(f"Simplified prompt: {result.simplified_prompt}")
     
     Get segmentation mask
     mask = result.segmentation_mask


## 📊 Performance Metrics

| Metric                 | Traditional Approach | EdgeSeg AI  | Improvement                |
| ---------------------- | -------------------- | ----------- | -------------------------- |
| **Peak Memory Usage**  | 12-16GB              | 4-6GB       | 60-70% reduction           |
| **Minimum GPU Memory** | 8GB                  | 4GB         | 50% reduction              |
| **Processing Time**    | 1 Minute          | 6-8 Minutes | Trade-off for efficiency   |
| **Segmentation Quality** | High               | High        | Maintained                 |

## 🙏 Acknowledgments

This project is inspired by the groundbreaking work in the **LLM-Seg paper**. While the original implementation prioritizes accuracy, our adaptation focuses on computational efficiency to make advanced segmentation accessible on resource-constrained hardware.

Hats off to the **LLM-Seg authors (Junchi Wang and Lei Ke from ETH Zurich)** for their exceptional contribution to the field.

## 🤝 Contributing

We welcome contributions! Whether it's bug fixes, new features, or documentation improvements, please feel free to open issues and submit pull requests.

- 🐛 Bug fixes
- ✨ New features
- 📖 Documentation improvements
- 🎨 UI/UX enhancements

## ⭐ Support

If you find this project helpful, your support would mean the world:

- **Give it a star ⭐**
- **Share your feedback**
- **Try it out and report any issues**

<br>
<p align="center">
  ---
  <br>
  <strong>Made with ❤️ & 🔥 for the community</strong>
  <br>
  <em>Bringing advanced AI capabilities to resource-constrained environments</em>
</p>

