import cv2
import numpy as np
import os
import requests
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

# Function to extract features from the image
def extract_features(image):
    image = cv2.resize(image, (128, 128))  # Resize image to 128x128
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load sample images and extract features
def load_data(directory):
    features = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image_features = extract_features(image)
                    features.append(image_features)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Load data
sample_images_dir = 'D:\\skin\\sample_images'
features, labels = load_data(sample_images_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Real-time monitoring
def real_time_monitoring():
    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        image_features = extract_features(frame)
        image_features = image_features.reshape(1, -1)
        prediction = knn.predict(image_features)[0]
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Predicted Disease: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-time Monitoring', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run real-time monitoring
real_time_monitoring()
