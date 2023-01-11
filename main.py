import cv2
import numpy as np

from keras.models import model_from_json


class EmotionRecognizer:

    def __init__(self):
        self.emotion_labels = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad",
                               6: "Surprised"}
        self._load_emotion_model()

    def _load_emotion_model(self) -> None:
        """Load the pre-trained model """
        with open('model/emotion_detection_model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)

        self.model.load_weights("model/emotion_detection_model.h5")

    def recognize_emotions(self, video_capture: cv2.VideoCapture) -> None:
        """Recognize emotions inside the frames of a video
        Args:
            video_capture: A cv2.VideoCapture object for the video to process
        """
        face_detector = cv2.CascadeClassifier('faceDetection/facedetector.xml')
        while True:
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_in_frame = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x_axis, y_axis, width, height) in faces_in_frame:
                cv2.rectangle(frame, (x_axis, y_axis - 50), (x_axis + width, y_axis + height + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y_axis:y_axis + height, x_axis:x_axis + width]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = self.model.predict(cropped_img)
                max_index = int(np.argmax(emotion_prediction))
                cv2.putText(frame, self.emotion_labels[max_index], (x_axis + 5, y_axis - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    emotion_recognizer = EmotionRecognizer()

    # start the webcam
    # video_capture = cv2.VideoCapture(0)

    # or use a video
    video_capture = cv2.VideoCapture("Videos/suprised.mp4")

    emotion_recognizer.recognize_emotions(video_capture)
