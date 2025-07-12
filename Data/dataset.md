# Dataset Documentation - EdgeSeg AI

## Dataset Overview

The EdgeSeg AI project utilizes a comprehensive dataset designed for training multimodal segmentation models with challenging prompt-based scenarios. This dataset combines custom annotations with established computer vision benchmarks to create a robust training environment for CPU-optimized AI models.

## Dataset Structure

### Core Dataset Components

| Component | Description | Format |
|-----------|-------------|---------|
| `dataset.xlsx` | Master dataset file containing image paths, prompts, and annotations | Excel Spreadsheet |
| `dataset_images/` | Consolidated image directory (post-processing) | Image Files |
| VOC Dataset | Additional training data from PASCAL VOC 2012 | Standard VOC Format |

### Dataset Schema

The `dataset.xlsx` file contains the following structured information:

- **Image Paths** - Customizable file paths for flexible dataset organization
- **Tricky Prompt Questions** - Complex natural language queries designed to challenge model understanding
- **Bounding Box Coordinates** - Precise label annotations for training and testing
- **Image Dimensions** - Width and height specifications for each image
- **Image Categories** - Classification labels for multi-class segmentation tasks

## Dataset Setup Instructions

### Step 1: Image Consolidation

Due to GitHub upload limitations, the image dataset is split across multiple archives:

