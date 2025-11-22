Teachable Machine - Multi-Class Image Classifier
Description

A Streamlit web app where users can upload images, define multiple classes, train machine learning models (Logistic Regression, Random Forest, CNN), and test predictions in real-time using uploaded or webcam images. Includes image validation and side-by-side prediction results.

Features

Upload multiple images per class

Train models: Logistic Regression, Random Forest, CNN

Real-time predictions for uploaded images or live webcam feed

Side-by-side comparison of model predictions

Image format and minimum class size validation

Modular file structure (data, trainers, models, inference, UI)

Folder Structure
project/
│
├─ data/           # Uploaded images / datasets
├─ trainers/       # Scripts for training LR, RF, CNN
├─ models/         # Saved model files (.pkl, .h5)
├─ inference/      # Prediction functions
└─ ui/             # Streamlit interface (app.py)
Requirements

Python 3.8+

Streamlit

TensorFlow / Keras

scikit-learn

PIL / numpy
