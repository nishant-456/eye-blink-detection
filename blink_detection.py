import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

# -------------------------------
# FUNCTION: Calculate EAR
# -------------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    return (A + B) / (2.0 * C)

# -------------------------------
# PARAMETERS (TUNED)
# -------------------------------
EAR_THRESHOLD = 0.22
FRAME_CHECK = 5

blink_count = 0
counter = 0
blink_cooldown = 0

# -------------------------------
# LOAD MODELS
# -------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR
        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye points
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # -------------------------------
        # BLINK LOGIC WITH COOLDOWN
        # -------------------------------
        if ear < EAR_THRESHOLD:
            counter += 1
        else:
            if counter >= FRAME_CHECK and blink_cooldown == 0:
                blink_count += 1
                blink_cooldown = 10  # prevent multiple counts
            counter = 0

        # Reduce cooldown
        if blink_cooldown > 0:
            blink_cooldown -= 1

        # -------------------------------
        # STATUS
        # -------------------------------
        status = "Open"
        if ear < EAR_THRESHOLD:
            status = "Closed"

        # -------------------------------
        # DISPLAY UI
        # -------------------------------
        cv2.putText(frame, "Eye Blink Detection System", (100, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Eye Blink Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()