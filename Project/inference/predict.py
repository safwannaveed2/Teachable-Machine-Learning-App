import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load models
lr_model = joblib.load("../models/lr_model.pkl")
rf_model = joblib.load("../models/rf_model.pkl")
cnn_model = load_model("../models/cnn_model.h5")

def predict_all(img_arr):
    # img_arr -> 128x128x3 numpy array
    img_flat = img_arr.reshape(1,-1)/255.0
    img_cnn = img_arr.reshape(1,128,128,3)/255.0

    pred_lr = lr_model.predict(img_flat)[0]
    pred_rf = rf_model.predict(img_flat)[0]
    pred_cnn = np.argmax(cnn_model.predict(img_cnn), axis=1)[0]

    return pred_lr, pred_rf, pred_cnn
