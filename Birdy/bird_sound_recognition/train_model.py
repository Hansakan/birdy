# train_model.py
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder

# Load features and labels
X = np.load('features.npy')
y = np.load('labels.npy')

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X to fit the model
X = X[..., np.newaxis]

# Build the model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_categorical, epochs=50, batch_size=32, validation_split=0.2)

# Save the model and label encoder
model.save('bird_sound_model.h5')
np.save('label_encoder.npy', label_encoder.classes_)
