from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # for saving models

def train_lr_rf(X, y):
    # Flatten if needed
    X_flat = X.reshape(len(X), -1)/255.0
    y_flat = y

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_val, lr_model.predict(X_val))

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_val, rf_model.predict(X_val))

    # Save models
    joblib.dump(lr_model, "../models/lr_model.pkl")
    joblib.dump(rf_model, "../models/rf_model.pkl")

    return lr_model, rf_model, lr_acc, rf_acc
joblib.dump(lr_model, "../models/lr_model.pkl")
joblib.dump(rf_model, "../models/rf_model.pkl")
