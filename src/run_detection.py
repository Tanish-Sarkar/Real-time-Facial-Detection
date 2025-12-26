import cv2 
from pathlib import Path

def is_blurry(gray_roi, thresold=80):
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    return laplacian_var, laplacian_var < thresold



def main():
    cascade_path = Path("assets/haarcascade_frontalface_default.xml")
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        print("Error: Haar cascade not loaded.")
        return
    
    print("Cascade loaded:", not face_cascade.empty())

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return    
    
    print("✅ Face detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        frame = cv2.resize(frame, (640, 480))
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60,60))

        # face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Global msg
        cv2.putText(
            frame,
            "Blur Detection Active",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # display 
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]

            blur_score, blurry = is_blurry(face_gray)

            if blurry:
                color = (0, 0, 255)
                label = "BLURRY"
            else:
                color = (0, 255, 0)
                label = "CLEAR"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label INSIDE box
            cv2.putText(
                frame,
                label,
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Draw blur score below label
            cv2.putText(
                frame,
                f"Score: {blur_score:.1f}",
                (x + 5, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Program exited cleanly.")

if __name__ == "__main__":
    main()