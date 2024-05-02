import cv2
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,  # Set to 1 to focus on a single face
    refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Access webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()

    # Convert frame to RGB format (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Face Mesh
    results = mp_face_mesh.process(rgb_frame)

    # Draw landmarks (if detections exist)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Display frame
    cv2.imshow('Webcam - Face Detection', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
