---
fine-tuned on base_model: microsoft/Florence-2-large
library_name: peft
---

# Model Card for Model ID

This model is Fine-tuned to predict bounding boxes on input images as per the simplified input prompt. 

## Model Details

### Model Description

This model is fine-tuned based on the Datset shared in the Github page named "Data". It uses pretrained Model Florence-2-large and with the help of LoRA (PEFT), it is finetuned to accurately predicting boxes. After training for 15 epochs we could bring the loss from 8.08 to ~0.1 which enhances the capability of predicting bounding box on the image with accurate co-ordinates that will help SAM for better Segmentation.



- **Developed by:** Rishab K Pattnaik
- **Funded by [optional]:** N/A
- **Shared by [optional]:** N/A
- **Model type:** PEFT Fine-Tuned model over microsoft/Florence-2-large.
- **Language(s) (NLP):** Coded in Python. Prefers input in form of Simple Natural Language.
- **Finetuned from model [optional]:** microsoft/Florence-2-large

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [(https://huggingface.co/microsoft/Florence-2-large)]
- **Paper [optional]:** [CVPR 2024 Conference Paper]

## Uses

You can use it in task where you want to predict bounding box with prompts. (Advanced YOLO that supports Natural Language Sentence)

### Direct Use

You can directly plug-n-play for your use cases. 

## Bias, Risks, and Limitations

Since its a project from a Student student without Funding, it has limited capabilites because of small dataset and lesser hardware. So it has learned to predict bounding box on limited objects which includes most common examples,but cannot predict niche examples. 
But, this idea can be used for much complex Multi-Models Architectures that will indeed change the scope and raise the bar of Computer Vision. You can fine-tune it for your use cases to extract exceptional results from this model and idea.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## How to Get Started with the Model

Just load the model file and set the path. Extract Processors and you are all set.


## Training Details

### Training Data

In Data Directory of this repo. 80% Train ; 5% Val ; 15% Test 

### Training Procedure

Peft LoRA Training with r = 64 and lora_alpha = 128. 
We also added Dropout of 0.1 with learning rate of 1e-4.
Num of Epochs = 15

#### Preprocessing [optional]

All image are set to dims (224,224,3). Gray Scale must be converted to RGB.

#### Training Env [optional]

Trained in Google-Colab T4 GPU (Free-Tier)

### Testing Data, Factors & Metrics

#### Testing Data

From Dataset.

#### Summary
Good Fined-Tuned model with its base as reserch-backed, performs very well given the size it comes in. Less training cost helps other developers to use the methodology to deploy in task-specific roles without the use of much hardware.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Colab T4 GPU
- **Hours used:** 30 Hours
- **Cloud Provider:** Google Colab
- **Compute Region:** India
- **Carbon Emitted:** N/A

- PEFT 0.15.2