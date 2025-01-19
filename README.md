# CholecT50 Triplet Detection Model

This repository contains code for training and predicting triplets in the CholecT50 dataset using a ResNet18 backbone. The model predicts the presence of 100 different triplets in surgical videos.

## Overview

The code is divided into two main parts:

1. **Training:** This section defines the model architecture, dataset loading, training loop, and saves the trained model weights. The core training logic is within the `main_train` function.
2. **Prediction:** This section loads a trained model and generates predictions on a set of videos, outputting the results in a JSON format. The prediction logic is within the `generate_predictions` function.

## Dataset

The code uses the CholecT50 dataset. The expected dataset structure is:

CholecT50/
├── videos/
│   ├── VID01/
│   │   ├── frame0001.jpg
│   │   ├── frame0002.jpg
│   │   ├── ...
│   ├── VID02/
│   │   ├── ...
│   ├── ...
│   └── VID111/
│       ├── ...
└── labels/
├── VID01.json
├── VID02.json
├── ...
└── VID111.json



Each `VIDXX.json` file contains triplet information for corresponding video frames.

## Model Architecture

The model utilizes a ResNet18 backbone pre-trained on ImageNet. Key modifications include:

- Freezing convolutional layers to retain pre-trained knowledge.
- Replacing the final fully connected layer with a custom triplet detection head.
- The custom head consists of layers for feature mapping, activation, dropout, and final triplet prediction.

## Usage

### Training

1. **Prepare the Dataset**: Organize CholecT50 dataset as specified.
2. **Install Dependencies**: `pip install torch torchvision Pillow tqdm`.
3. **Run Training**: Execute the training script to save model weights (`trained_model.pth`).

### Prediction

1. **Ensure Trained Model**: Have `trained_model.pth` available.
2. **Set Dataset Path**: Update `ROOT_DIR` in prediction script to dataset directory.
3. **Run Prediction**: Execute prediction script to output JSON format predictions.

## Code Explanation

- **Dataset Class (`CholecT50Dataset`)**: Loads dataset images and labels.
- **Model (`TripletDetectionModel`)**: Combines ResNet18 backbone with custom triplet head.
- **Training (`train_model`)**: Iterates through training data, calculates loss, and updates model.
- **Prediction (`generate_predictions`)**: Loads model, predicts on test data, and formats output to JSON.

## Output Format

Predictions are formatted and printed to standard output in JSON format, organized by video and frame with triplet probabilities and detection data (if available).

## Dependencies

- Python 3
- PyTorch
- Torchvision
- Pillow (PIL)
- tqdm

## Further Improvements

- Implement bounding box regression with IoU or Smooth L1 loss.
- Explore different backbone architectures (e.g., ResNet50, EfficientNet).
- Incorporate data augmentation for improved model generalization.
- Introduce evaluation metrics like mean Average Precision (mAP).
- Add a configuration file for managing hyperparameters.
