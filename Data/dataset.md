📊 Dataset Documentation - EdgeSeg AI
🎯 Dataset Overview
The EdgeSeg AI project utilizes a comprehensive dataset designed for training multimodal segmentation models with challenging prompt-based scenarios. This dataset combines custom annotations with established computer vision benchmarks to create a robust training environment for CPU-optimized AI models.

📁 Dataset Structure
Core Dataset Components
Component	Description	Format
dataset.xlsx	Master dataset file containing image paths, prompts, and annotations	Excel Spreadsheet
dataset_images/	Consolidated image directory (post-processing)	Image Files
VOC Dataset	Additional training data from PASCAL VOC 2012	Standard VOC Format
Dataset Schema
The dataset.xlsx file contains the following structured information:

📍 Image Paths - Customizable file paths for flexible dataset organization

🎯 Tricky Prompt Questions - Complex natural language queries designed to challenge model understanding

📦 Bounding Box Coordinates - Precise label annotations for training and testing

📐 Image Dimensions - Width and height specifications for each image

🏷️ Image Categories - Classification labels for multi-class segmentation tasks

🔧 Dataset Setup Instructions
Step 1: Image Consolidation
Due to GitHub upload limitations, the image dataset is split across multiple archives:

bash
# Download and extract both archives
unzip dataset_images.zip
unzip dataset_images2.zip

# Consolidate into single directory
mkdir dataset_images
mv dataset_images_part1/* dataset_images/
mv dataset_images_part2/* dataset_images/
Step 2: VOC Dataset Integration
Enhance your dataset with PASCAL VOC 2012 benchmark data:

bash
# Download PASCAL VOC 2012 dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract dataset
tar -xf VOCtrainval_11-May-2012.tar
Step 3: Dataset Verification
Ensure proper dataset structure:

text
EdgeSeg-AI/
├── dataset.xlsx
├── dataset_images/          # Consolidated image directory
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── VOCdevkit/              # PASCAL VOC 2012 data
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
🎨 Custom Dataset Creation
Creating Your Own Dataset
To adapt this framework for your specific use case:

Prepare Images - Organize your image collection

Create Annotations - Generate bounding box coordinates

Design Prompts - Develop challenging natural language queries

Update Excel File - Modify dataset.xlsx with your data

Validate Structure - Ensure consistency with the provided schema

Dataset Best Practices
Diverse Prompts - Include varied linguistic patterns and complexity levels

Balanced Categories - Maintain representative samples across all classes

Quality Annotations - Ensure precise bounding box coordinates

Scalable Organization - Use consistent naming conventions and directory structure

🚀 Training Integration
This dataset structure is specifically designed for:

Multimodal Learning - Combining visual and textual understanding

Prompt-Based Training - Teaching models to respond to natural language queries

Edge Deployment - Optimized for CPU-only inference scenarios

Robust Evaluation - Comprehensive testing across diverse scenarios

💡 Key Features
Innovative Prompt Design
The dataset includes tricky prompt questions that challenge traditional computer vision approaches, enabling the development of more sophisticated multimodal AI systems.

Flexible Architecture
The modular dataset structure allows for easy customization and extension, supporting various research and production scenarios.

Production-Ready Format
Designed with deployment considerations in mind, ensuring compatibility with resource-constrained environments and CPU-only inference requirements.

This dataset framework supports the EdgeSeg AI project's mission to democratize advanced computer vision capabilities through CPU-optimized, multimodal segmentation technology.
