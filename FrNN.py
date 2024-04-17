import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def generate_face_mesh(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    else:
        return None

def extract_facial_features(face_landmarks):
    # Your feature extraction code here
    pass

def load_dataset(dataset_dir):
    X, y = [], []
    label_encoder = LabelEncoder()
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(person_dir, filename)
                    image = cv2.imread(image_path)
                    # Resize images to 250x250 if not already
                    if image.shape[:2] != (250, 250):
                        image = cv2.resize(image, (250, 250))
                    # Convert BGR image to RGB (Mediapipe requires RGB input)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Detect face mesh
                    face_landmarks = generate_face_mesh(image_rgb)
                    if face_landmarks:
                        features = extract_facial_features(face_landmarks)
                        X.append(features)
                        y.append(person_name)  
    X = np.array(X)
    y = label_encoder.fit_transform(y)  # Encode class labels into numeric format
    return X, y

def create_complex_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    # Load dataset
    dataset_dir = "Dataset/"
    X, y = load_dataset(dataset_dir)
    num_classes = len(np.unique(y))
    input_shape = (len(X[0]),)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for CNN
    X_train = X_train.reshape(-1, 250, 250, 1)
    X_test = X_test.reshape(-1, 250, 250, 1)

    # Define model
    model = create_complex_model(input_shape, num_classes)
    
    # Compile model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Save model
    model.save("face_recognition_model_complex.h5")
