from tensorflow.keras import layers, models

def train_cnn(X, y, num_classes):
    X_cnn = X.astype('float32') / 255.0

    cnn_model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_cnn, y, epochs=5, validation_split=0.2)

    cnn_model.save("../models/cnn_model.h5")
    return cnn_model
cnn_model.save("../models/cnn_model.h5")
