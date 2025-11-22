import streamlit as st
from PIL import Image
import numpy as np
from trainers.train_lr_rf import train_lr_rf
from trainers.train_cnn import train_cnn
from inference.predict import predict_all

st.title("🖼️ Teachable Machine - Modular App")

# --- Class definition & image upload ---
# (same as before, includes class_images dictionary, load_image, X, y)

# --- Train Models Button ---
if st.button("Train Models"):
    lr_model, rf_model, lr_acc, rf_acc = train_lr_rf(X, y)
    cnn_model = train_cnn(X, y, num_classes=len(class_images))
    st.write(f"LR Accuracy: {lr_acc:.2f}, RF Accuracy: {rf_acc:.2f}")
    st.success("✅ Models Trained!")

# --- Prediction Section ---
test_file = st.file_uploader("Upload or capture image", type=["jpg","png","jpeg"])
if test_file:
    img = Image.open(test_file).convert('RGB').resize((128,128))
    st.image(img, width=150)
    img_arr = np.array(img)
    pred_lr, pred_rf, pred_cnn = predict_all(img_arr)
    class_names = list(class_images.keys())
    st.write(f"LR: {class_names[pred_lr]}, RF: {class_names[pred_rf]}, CNN: {class_names[pred_cnn]}")
