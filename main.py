import cv2
import numpy as np
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Specify the directory containing reference images
reference_images_dir = 'reference_images'

# Get a list of reference image filenames
reference_image_files = os.listdir(reference_images_dir)

# Load all reference images and convert them to grayscale
reference_images = []
for filename in reference_image_files:
    if filename.endswith('.jpg') or filename.endswith('.png'):
        reference_image = cv2.imread(os.path.join(reference_images_dir, filename))
        reference_images.append(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))

# Create a window to display the camera feed
cv2.namedWindow('Facial Recognition')

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera or specify a camera index

# Flag to indicate whether to capture a new reference image
capture_new_reference = False

while True:
    # Capture video from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture video.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the camera feed
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the detected face region from the frame
        detected_face = gray[y:y+h, x:x+w]

        # Resize the detected face to match the reference image sizes
        detected_face = cv2.resize(detected_face, (reference_images[0].shape[1], reference_images[0].shape[0]))

        if capture_new_reference:
            # Save the detected face as a new reference image
            reference_image_filename = f'new_reference_{len(reference_images)}.jpg'
            cv2.imwrite(os.path.join(reference_images_dir, reference_image_filename), detected_face)
            print(f'Saved new reference image: {reference_image_filename}')
            reference_images.append(detected_face)
            capture_new_reference = False

        else:
            # Compare the detected face with each reference image
            match_found = False
            for i, reference_image in enumerate(reference_images):
                match_score = cv2.matchTemplate(reference_image, detected_face, cv2.TM_CCOEFF_NORMED)
                match_threshold = 0.8  # Adjust this threshold as needed

                if np.max(match_score) >= match_threshold:
                    match_found = True
                    break  # Break out of the loop if a match is found

            if match_found:
                cv2.putText(frame, 'Face Matched', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the camera feed with face detection and matching
    cv2.imshow('Facial Recognition', frame)

    # Press 's' to capture a new reference image
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        capture_new_reference = True

    # Press 'q' to exit the program
    if key & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
