import cv2
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, render_template_string
import os

# ------------------ CONFIG ------------------

CASCADE_PATH = Path("assets/haarcascade_frontalface_default.xml")
SAVE_INTERVAL = 5
BLUR_THRESHOLD = 80

# ------------------ SETUP ------------------

app = Flask(__name__)

os.makedirs("saved_frames", exist_ok=True)
os.makedirs("logs", exist_ok=True)


face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))


last_saved_time = 0

logging.basicConfig(
    filename="logs/detections.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# ------------------ HELPERS ------------------

def is_blurry(gray_roi, threshold=BLUR_THRESHOLD):
    score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return score, score < threshold


def generate_frames():
    global last_saved_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame read failed")
                break

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
            )

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                blur_score, blurry = is_blurry(face_gray)

                color = (0, 0, 255) if blurry else (0, 255, 0)
                label = "BLURRY" if blurry else "CLEAR"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x + 5, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Score: {blur_score:.1f}",
                            (x + 5, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                now = datetime.now().timestamp()
                if blurry and (now - last_saved_time) > SAVE_INTERVAL:
                    filename = f"saved_frames/blur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    logging.info(f"BLUR_DETECTED | score={blur_score:.2f}")
                    last_saved_time = now

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()
        print("📷 Camera released")


# ------------------ ROUTES ------------------

@app.route("/")
def index():
    return render_template_string(
        """
        <html>
            <head><title>Live CV Stream</title></head>
            <body>
                <h2>Real-Time Face & Blur Detection</h2>
                <img src="/video_feed" width="640" />
            </body>
        </html>
        """
    )

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

