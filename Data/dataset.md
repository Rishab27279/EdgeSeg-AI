# EdgeSeg-AI Dataset

A comprehensive dataset for training multimodal segmentation models with challenging prompt-based scenarios, optimized for CPU deployment.

## Overview

The EdgeSeg AI project utilizes a comprehensive dataset designed for training multimodal segmentation models with challenging prompt-based scenarios. This dataset combines custom annotations with established computer vision benchmarks to create a robust training environment for CPU-optimized AI models.

## Features

- **Multimodal Learning** - Combines visual and textual understanding
- **Prompt-Based Training** - Natural language query responses
- **Edge Deployment** - CPU-optimized inference
- **Flexible Architecture** - Easy customization and extension
- **Production-Ready** - Resource-constrained environment compatibility

## Dataset Structure

### Core Components

| Component | Description | Format |
|-----------|-------------|---------|
| `dataset.xlsx` | Master dataset file containing image paths, prompts, and annotations | Excel Spreadsheet |
| `dataset_images/` | Consolidated image directory (post-processing) | Image Files |
| `VOC Dataset` | Additional training data from PASCAL VOC 2012 | Standard VOC Format |

### Schema

The `dataset.xlsx` file contains:

- **Image Paths** - Customizable file paths for flexible dataset organization
- **Tricky Prompt Questions** - Complex natural language queries designed to challenge model understanding
- **Bounding Box Coordinates** - Precise label annotations for training and testing
- **Image Dimensions** - Width and height specifications for each image
- **Image Categories** - Classification labels for multi-class segmentation tasks

## Quick Start

### Prerequisites

- Python 3.7+
- Git
- wget (for Linux/macOS) or equivalent download tool

### Installation

1. **Clone the repository**
        git clone https://github.com/your-username/EdgeSeg-AI.git
        cd EdgeSeg-AI

2. **Download and consolidate images**
   
     Download and extract both archives ~
        `dataset_images.zip` and 
        `dataset_images2.zip`
    
    Consolidate into single directory ~
        `mkdir dataset_images`

4. **Integrate VOC Dataset** (Optional)
        Download PASCAL VOC 2012 dataset ~
        `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar` and 
        Extract dataset ~
        `tar -xf VOCtrainval_11-May-2012.tar`

### Verify Installation

Ensure your directory structure looks like this:

    EdgeSeg-AI/
    ├── dataset.xlsx
    ├── dataset_images/ # Consolidated image directory
    │ ├── image_001.jpg
    │ ├── image_002.jpg
    │ └── ...
    └── VOCdevkit/ # PASCAL VOC 2012 data (optional)
    └── VOC2012/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/

## Usage

### Creating Custom Datasets

To adapt this framework for your specific use case:

1. **Prepare Images** - Organize your image collection
2. **Create Annotations** - Generate bounding box coordinates
3. **Design Prompts** - Develop challenging natural language queries
4. **Update Excel File** - Modify `dataset.xlsx` with your data
5. **Validate Structure** - Ensure consistency with the provided schema

### Best Practices

- **Diverse Prompts** - Include varied linguistic patterns and complexity levels
- **Balanced Categories** - Maintain representative samples across all classes
- **Quality Annotations** - Ensure precise bounding box coordinates
- **Scalable Organization** - Use consistent naming conventions and directory structure

## Key Features

### Innovative Prompt Design

The dataset includes tricky prompt questions that challenge traditional computer vision approaches, enabling the development of more sophisticated multimodal AI systems.

### Flexible Architecture

The modular dataset structure allows for easy customization and extension, supporting various research and production scenarios.

### Production-Ready Format

Designed with deployment considerations in mind, ensuring compatibility with resource-constrained environments and CPU-only inference requirements.

## Training Integration

This dataset structure is specifically designed for:

- **Multimodal Learning** - Combining visual and textual understanding
- **Prompt-Based Training** - Teaching models to respond to natural language queries
- **Edge Deployment** - Optimized for CPU-only inference scenarios
- **Robust Evaluation** - Comprehensive testing across diverse scenarios

## Support

If you encounter any issues or have questions, please open an issue on GitHub or feel free to contact me.

---

This dataset framework supports the EdgeSeg AI project's mission to democratize advanced computer vision capabilities through CPU-optimized, multimodal segmentation technology.
