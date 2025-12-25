import cv2 
import pathlib as Path

def main():

    cascade_path = Path("assets/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        print("Error: Haar cascade not loaded.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("✅ Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from webcam.")
            break

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Program exited cleanly.")

if __name__ == "__main__":
    main()