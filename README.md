# Real-Time Face Detection and Tracking System
A deep learning-based face detection system that performs real-time face localization using a custom CNN architecture with VGG16 backbone. The model predicts both face presence (classification) and bounding box coordinates (regression) for accurate face tracking.
Overview
This project implements a dual-output neural network that simultaneously:

Classifies whether a face is present in the frame
Localizes the face by predicting bounding box coordinates

The system uses **transfer learning with VGG16** as the feature extractor and **custom dense layers for classification and regression tasks.**

## Features -
- Real-time face detection via webcam feed
- Custom data augmentation pipeline using Albumentations
- Dual-task learning with combined classification and localization loss
- VGG16 backbone for robust feature extraction
- Live bounding box visualization on detected faces

## Architecture -
graph TD
    A[Input Image<br/>120x120x3] --> B[VGG16 Backbone<br/>Pretrained Feature Extractor]
    B --> C[Global Max Pooling 2D]
    C --> D[Classification Branch]
    C --> E[Regression Branch]
    
    D --> D1[Dense Layer<br/>2048 units, ReLU]
    D1 --> D2[Dense Layer<br/>1 unit, Sigmoid]
    D2 --> D3[Output: Face Presence<br/>0 = No Face, 1 = Face]
    
    E --> E1[Dense Layer<br/>2048 units, ReLU]
    E1 --> E2[Dense Layer<br/>4 units, Sigmoid]
    E2 --> E3[Output: Bounding Box<br/>x_min, y_min, x_max, y_max]
    
    style A fill:#e1f5ff
    style D3 fill:#d4edda
    style E3 fill:#d4edda
    style B fill:#fff3cd
