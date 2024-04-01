import tkinter as tk
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image, ImageTk

class DrowsinessDetector:
    def __init__(self, window, video_path):
        self.window = window
        self.window.title("Drowsiness Detection")
        self.video_path = video_path
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.model = self.load_model()

        self.video_capture = cv2.VideoCapture(self.video_path)
        self.detect_drowsiness()

    def load_model(self):
        # Load the model architecture from JSON file
        json_file = open('eye_state_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Load weights into new model
        loaded_model.load_weights("eye_state_model_weights.h5")
        print("Loaded model from disk")
        return loaded_model

    def detect_drowsiness(self):
        ret, frame = self.video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_img = cv2.resize(eye_img, (48, 48))
                    eye_img = np.expand_dims(eye_img, axis=-1)
                    eye_img = np.expand_dims(eye_img, axis=0)
                    prediction = self.model.predict(eye_img)
                    eye_state = "Open" if prediction[0][0] > 0.5 else "Closed"
                    # Draw filled rectangle as background
                    cv2.rectangle(frame, (x, y - 30), (x + 200, y - 10), (0, 255, 255), -1)
                    # Draw text on top of the rectangle with larger font size
                    cv2.putText(frame, f"Eye State: {eye_state}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if eye_state == "Closed":
                        cv2.putText(frame, "Drowsiness Alert!", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = cv2.resize(self.photo, (640, 480))
            self.photo = Image.fromarray(self.photo)
            self.photo = ImageTk.PhotoImage(image=self.photo)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.window.after(10, self.detect_drowsiness)

def main():
    root = tk.Tk()
    video_path = "drowsy2.mp4"  
    app = DrowsinessDetector(root, video_path)
    root.mainloop()

if __name__ == "__main__":
    main()
