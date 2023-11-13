# Dual-Modal Deep Belief Network with Fusion

This repository contains an implementation of a Dual-Modal Deep Belief Network (DBN) with a fusion mechanism for handling dual-modal data. The network is designed to work with two different modalities of data and fuse them to perform a classification task.

## Overview

- The network architecture consists of the following components:
  - Dual-Modal Deep Belief Network (DBN): It consists of two Restricted Boltzmann Machines (RBMs) with attention mechanisms for each modality. These RBMs handle the feature extraction and interaction between modalities.
  - Encoders: Two different encoders are used for encoding data from each modality. One is based on LSTM, and the other is based on a 1D ResNet or Transformer Encoder.
  - Fusion Block: This block fuses the encoded representations from both modalities.
  - Fully Connected Layers: These layers perform the final classification task.

## Usage

1. **Installation**: Make sure you have the required libraries installed by running `pip install -r requirements.txt`.

2. **Training**: You can train the network using your dataset by modifying the training script. You may need to adjust hyperparameters, data loading, and preprocessing according to your specific data.

3. **Inference**: After training, you can use the trained model for inference tasks. Load the saved model weights and pass your data through the network to obtain predictions.

## Example Usage
Is in the bottom of nn_models.py file.

## Requirements

Python 3.7+
PyTorch
torchvision
numpy


## Acknowledgments
This project is based on the concept of Dual-Modal Deep Belief Networks (DBNs) with attention mechanisms.

## License
This project is licensed under the MIT License.

## Author

Zihong.Luo

