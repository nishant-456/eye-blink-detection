# 👁 Eye Blink Detection System

A real-time Eye Blink Detection System developed using Python, OpenCV, and Dlib. The application detects facial landmarks, identifies eye blinks using the Eye Aspect Ratio (EAR) algorithm, and displays the blink count in real time through a webcam.

---

## Features

- Real-time face detection
- Eye blink detection using facial landmarks
- Blink counter
- Live eye status detection (Open/Closed)
- Webcam-based monitoring

---

## Technologies Used

- Python
- OpenCV
- Dlib
- NumPy
- SciPy
- Imutils

---

## Project Structure

```
eye-blink-detection/
│── blink_detection.py
│── requirements.txt
│── .gitignore
└── README.md
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/eye-blink-detection.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Download the pretrained facial landmark model:

`shape_predictor_68_face_landmarks.dat`

4. Place the model file in the project directory.

5. Run the application

```bash
python blink_detection.py
```

---

## Future Enhancements

- Driver drowsiness detection
- Fatigue monitoring
- Multi-face support
- Performance optimization

---

## Author

**Nishant Kirar**

Final Year B.Tech (Artificial Intelligence & Robotics)  
Madhav Institute of Technology and Science (MITS), Gwalior
