import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Define the FaceMeshDetector class
class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def draw_face_mesh(self, frame, face_landmarks):
        # Draw different components of the face mesh
        if self.mp_face_mesh.FACEMESH_TESSELATION:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.filter_landmark_indices(self.mp_face_mesh.FACEMESH_TESSELATION, len(face_landmarks.landmark)),
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        if self.mp_face_mesh.FACEMESH_CONTOURS:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.filter_landmark_indices(self.mp_face_mesh.FACEMESH_CONTOURS, len(face_landmarks.landmark)),
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        if self.mp_face_mesh.FACEMESH_IRISES:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.filter_landmark_indices(self.mp_face_mesh.FACEMESH_IRISES, len(face_landmarks.landmark)),
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    
    def filter_landmark_indices(self, connections, num_landmarks):
        filtered_connections = []
        for connection in connections:
            start_idx, end_idx = connection
            if 0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks:
                filtered_connections.append(connection)
        return filtered_connections


#   Load the trained model
model = load_model("face_recognition_model.h5")

# Initialize the FaceMeshDetector
detector = FaceMeshDetector()

# Function to generate face mesh using MediaPipe
def generate_face_mesh(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results

# Function to extract facial features
# def extract_facial_features(face_landmarks):
#     feature_vector = []
    
#     # Example: Compute distances between nose and other facial landmarks
#     nose_tip = face_landmarks.landmark[6]  # Index 6 corresponds to the tip of the nose
#     left_eye_inner = face_landmarks.landmark[133]  # Index 133 corresponds to the inner corner of the left eye
#     left_eye_outer = face_landmarks.landmark[155]  # Index 155 corresponds to the outer corner of the left eye
#     right_eye_inner = face_landmarks.landmark[362]  # Index 362 corresponds to the inner corner of the right eye
#     right_eye_outer = face_landmarks.landmark[382]  # Index 382 corresponds to the outer corner of the right eye
#     mouth_left = face_landmarks.landmark[61]  # Index 61 corresponds to the left corner of the mouth
#     mouth_right = face_landmarks.landmark[291]  # Index 291 corresponds to the right corner of the mouth
    
#     # Distances between nose and eye corners
#     nose_left_eye_inner_distance = np.linalg.norm([nose_tip.x - left_eye_inner.x, nose_tip.y - left_eye_inner.y, nose_tip.z - left_eye_inner.z])
#     nose_left_eye_outer_distance = np.linalg.norm([nose_tip.x - left_eye_outer.x, nose_tip.y - left_eye_outer.y, nose_tip.z - left_eye_outer.z])
#     nose_right_eye_inner_distance = np.linalg.norm([nose_tip.x - right_eye_inner.x, nose_tip.y - right_eye_inner.y, nose_tip.z - right_eye_inner.z])
#     nose_right_eye_outer_distance = np.linalg.norm([nose_tip.x - right_eye_outer.x, nose_tip.y - right_eye_outer.y, nose_tip.z - right_eye_outer.z])
    
#     feature_vector.extend([nose_left_eye_inner_distance, nose_left_eye_outer_distance, nose_right_eye_inner_distance, nose_right_eye_outer_distance])
    
#     # Distances between eyes
#     left_eye_distance = np.linalg.norm([left_eye_inner.x - left_eye_outer.x, left_eye_inner.y - left_eye_outer.y, left_eye_inner.z - left_eye_outer.z])
#     right_eye_distance = np.linalg.norm([right_eye_inner.x - right_eye_outer.x, right_eye_inner.y - right_eye_outer.y, right_eye_inner.z - right_eye_outer.z])
    
#     feature_vector.extend([left_eye_distance, right_eye_distance])
    
#     # Distance between mouth corners
#     mouth_distance = np.linalg.norm([mouth_left.x - mouth_right.x, mouth_left.y - mouth_right.y, mouth_left.z - mouth_right.z])
#     feature_vector.append(mouth_distance)
    
#     # Additional features: Angles, Ratios, Curvature, Symmetry, etc.
#     # Example:
#     # Angle between eyes
#     angle_eyes = angle_between_vectors([left_eye_inner.x, left_eye_inner.y, left_eye_inner.z], [left_eye_outer.x, left_eye_outer.y, left_eye_outer.z], [right_eye_inner.x, right_eye_inner.y, right_eye_inner.z])
#     feature_vector.append(angle_eyes)
    
#     # Ratio of eye distances to mouth distance
#     eye_to_mouth_ratio = (left_eye_distance + right_eye_distance) / mouth_distance
#     feature_vector.append(eye_to_mouth_ratio)
    
#     # Ratio of eye distances to face width
#     face_width = np.linalg.norm([left_eye_outer.x - right_eye_outer.x, left_eye_outer.y - right_eye_outer.y, left_eye_outer.z - right_eye_outer.z])
#     eye_to_face_width_ratio = (left_eye_distance + right_eye_distance) / face_width
#     feature_vector.append(eye_to_face_width_ratio)
    
#     # Symmetry: Distance between left and right eye corners
#     eye_symmetry = np.linalg.norm([left_eye_outer.x - right_eye_outer.x, left_eye_outer.y - right_eye_outer.y, left_eye_outer.z - right_eye_outer.z])
#     feature_vector.append(eye_symmetry)
    
#     # Ratio of eye distances to nose width
#     nose_width = np.linalg.norm([nose_tip.x - face_landmarks.landmark[2].x, nose_tip.y - face_landmarks.landmark[2].y, nose_tip.z - face_landmarks.landmark[2].z])
#     eye_to_nose_width_ratio = (left_eye_distance + right_eye_distance) / nose_width
#     feature_vector.append(eye_to_nose_width_ratio)
    
#     # Curvature: Angle between nose bridge and mouth corners
#     angle_nose_mouth = angle_between_vectors([nose_tip.x, nose_tip.y, nose_tip.z], [mouth_left.x, mouth_left.y, mouth_left.z], [mouth_right.x, mouth_right.y, mouth_right.z])
#     feature_vector.append(angle_nose_mouth)
    
#     # Distances between nose and eyebrows
#     left_eyebrow_center = face_landmarks.landmark[70]  # Index 70 corresponds to the center of the left eyebrow
#     right_eyebrow_center = face_landmarks.landmark[336]  # Index 336 corresponds to the center of the right eyebrow
    
#     nose_left_eyebrow_distance = np.linalg.norm([nose_tip.x - left_eyebrow_center.x, nose_tip.y - left_eyebrow_center.y, nose_tip.z - left_eyebrow_center.z])
#     nose_right_eyebrow_distance = np.linalg.norm([nose_tip.x - right_eyebrow_center.x, nose_tip.y - right_eyebrow_center.y, nose_tip.z - right_eyebrow_center.z])
    
#     feature_vector.extend([nose_left_eyebrow_distance, nose_right_eyebrow_distance])
    
#     # Distance between eyes and eyebrows
#     left_eye_eyebrow_distance = np.linalg.norm([left_eye_inner.x - left_eyebrow_center.x, left_eye_inner.y - left_eyebrow_center.y, left_eye_inner.z - left_eyebrow_center.z])
#     right_eye_eyebrow_distance = np.linalg.norm([right_eye_inner.x - right_eyebrow_center.x, right_eye_inner.y - right_eyebrow_center.y, right_eye_inner.z - right_eyebrow_center.z])
    
#     feature_vector.extend([left_eye_eyebrow_distance, right_eye_eyebrow_distance])
    
#     # Ratio of nose width to eye distances
#     nose_width_eye_distance_ratio = nose_width / (left_eye_distance + right_eye_distance)
#     feature_vector.append(nose_width_eye_distance_ratio)
    
#     # Ratio of eye distances to face height
#     face_height = np.linalg.norm([left_eye_outer.x - left_eye_outer.y, left_eye_outer.y - left_eye_outer.y, left_eye_outer.z - left_eye_outer.y])
#     eye_to_face_height_ratio = (left_eye_distance + right_eye_distance) / face_height
#     feature_vector.append(eye_to_face_height_ratio)
    
#     # Ratio of eye distances to face area
#     face_area = face_width * face_height
#     eye_to_face_area_ratio = (left_eye_distance + right_eye_distance) / face_area
#     feature_vector.append(eye_to_face_area_ratio)
#     # Add more features as needed
    
#     return feature_vector
# Function to extract facial features
def extract_facial_features(face_landmarks):
    feature_vector = []
    
    # Example: Compute distances between nose and other facial landmarks
    nose_tip = face_landmarks.landmark[6]  # Index 6 corresponds to the tip of the nose
    left_eye_inner = face_landmarks.landmark[133]  # Index 133 corresponds to the inner corner of the left eye
    left_eye_outer = face_landmarks.landmark[155]  # Index 155 corresponds to the outer corner of the left eye
    right_eye_inner = face_landmarks.landmark[362]  # Index 362 corresponds to the inner corner of the right eye
    right_eye_outer = face_landmarks.landmark[382]  # Index 382 corresponds to the outer corner of the right eye
    mouth_left = face_landmarks.landmark[61]  # Index 61 corresponds to the left corner of the mouth
    mouth_right = face_landmarks.landmark[291]  # Index 291 corresponds to the right corner of the mouth
    
    # Distances between nose and eye corners
    nose_left_eye_inner_distance = np.linalg.norm([nose_tip.x - left_eye_inner.x, nose_tip.y - left_eye_inner.y, nose_tip.z - left_eye_inner.z])
    nose_left_eye_outer_distance = np.linalg.norm([nose_tip.x - left_eye_outer.x, nose_tip.y - left_eye_outer.y, nose_tip.z - left_eye_outer.z])
    nose_right_eye_inner_distance = np.linalg.norm([nose_tip.x - right_eye_inner.x, nose_tip.y - right_eye_inner.y, nose_tip.z - right_eye_inner.z])
    nose_right_eye_outer_distance = np.linalg.norm([nose_tip.x - right_eye_outer.x, nose_tip.y - right_eye_outer.y, nose_tip.z - right_eye_outer.z])
    
    feature_vector.extend([nose_left_eye_inner_distance, nose_left_eye_outer_distance, nose_right_eye_inner_distance, nose_right_eye_outer_distance])
    
    # Distances between eyes
    left_eye_distance = np.linalg.norm([left_eye_inner.x - left_eye_outer.x, left_eye_inner.y - left_eye_outer.y, left_eye_inner.z - left_eye_outer.z])
    right_eye_distance = np.linalg.norm([right_eye_inner.x - right_eye_outer.x, right_eye_inner.y - right_eye_outer.y, right_eye_inner.z - right_eye_outer.z])
    
    feature_vector.extend([left_eye_distance, right_eye_distance])
    
    # Distance between mouth corners
    mouth_distance = np.linalg.norm([mouth_left.x - mouth_right.x, mouth_left.y - mouth_right.y, mouth_left.z - mouth_right.z])
    feature_vector.append(mouth_distance)
    
    # Angle between eyes
    angle_eyes = angle_between_vectors([left_eye_inner.x, left_eye_inner.y, left_eye_inner.z], [left_eye_outer.x, left_eye_outer.y, left_eye_outer.z], [right_eye_inner.x, right_eye_inner.y, right_eye_inner.z])
    feature_vector.append(angle_eyes)
    
    # Ratio of eye distances to mouth distance
    eye_to_mouth_ratio = (left_eye_distance + right_eye_distance) / mouth_distance
    feature_vector.append(eye_to_mouth_ratio)
    
    # Ratio of eye distances to face width
    face_width = np.linalg.norm([left_eye_outer.x - right_eye_outer.x, left_eye_outer.y - right_eye_outer.y, left_eye_outer.z - right_eye_outer.z])
    eye_to_face_width_ratio = (left_eye_distance + right_eye_distance) / face_width
    feature_vector.append(eye_to_face_width_ratio)
    
    # Symmetry: Distance between left and right eye corners
    eye_symmetry = np.linalg.norm([left_eye_outer.x - right_eye_outer.x, left_eye_outer.y - right_eye_outer.y, left_eye_outer.z - right_eye_outer.z])
    feature_vector.append(eye_symmetry)
    
    # Ratio of nose width to eye distances
    nose_width = np.linalg.norm([nose_tip.x - face_landmarks.landmark[2].x, nose_tip.y - face_landmarks.landmark[2].y, nose_tip.z - face_landmarks.landmark[2].z])
    eye_to_nose_width_ratio = (left_eye_distance + right_eye_distance) / nose_width
    feature_vector.append(eye_to_nose_width_ratio)
    
    # Curvature: Angle between nose bridge and mouth corners
    angle_nose_mouth = angle_between_vectors([nose_tip.x, nose_tip.y, nose_tip.z], [mouth_left.x, mouth_left.y, mouth_left.z], [mouth_right.x, mouth_right.y, mouth_right.z])
    feature_vector.append(angle_nose_mouth)
    
    return feature_vector

def angle_between_vectors(p1, p2, p3):
    vector1 = np.array(p1) - np.array(p2)
    vector2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (norm_vector1 * norm_vector2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)



# Function to predict identity using the trained model
def predict_identity(features):
    # Preprocess features if needed
    # Ensure features are in the correct shape for prediction
    features = np.array(features)  # Convert to numpy array
    if len(features.shape) == 1:
        features = np.expand_dims(features, axis=0)  # Add batch dimension if missing
    # Make prediction using the trained model
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

# Get directory names as identity names
identity_names = os.listdir("Dataset/")

# Main function to process live video
def process_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and extract facial features
        results = generate_face_mesh(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                features = extract_facial_features(face_landmarks)
                
                # Predict identity
                predicted_class = predict_identity([features])
                
                # Draw face mesh
                detector.draw_face_mesh(frame, face_landmarks)
                
                # Draw bounding box and label
                left, top, right, bottom = get_bounding_box(face_landmarks, frame.shape[:2])
                identity_name = identity_names[predicted_class[0]]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, identity_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to calculate bounding box coordinates
def get_bounding_box(face_landmarks, frame_shape):
    landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
    landmarks[:, 0] *= frame_shape[1]
    landmarks[:, 1] *= frame_shape[0]
    left = int(np.min(landmarks[:, 0]))
    top = int(np.min(landmarks[:, 1]))
    right = int(np.max(landmarks[:, 0]))
    bottom = int(np.max(landmarks[:, 1]))
    return left, top, right, bottom

if __name__ == "__main__":
    process_video()
