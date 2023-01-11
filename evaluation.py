from typing import Generator
from keras.preprocessing.image import ImageDataGenerator
from keras.saving.legacy.model_config import model_from_json


class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.validation_data_gen = ImageDataGenerator(rescale=1. / 255)

    def _load_emotion_model(self) -> None:
        """Loads the emotion detection model from the 'model' directory.
        """
        with open('model/emotion_detection_model.json', 'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights('model/emotion_detection_model.h5')

    def _compile_emotion_model(self) -> None:
        """Compiles the emotion detection model.
        """
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def _create_validation_generator(self) -> Generator:
        """Creates a generator for generating validation data.
            Returns:
                A generator for generating validation data.
        """
        validation_data_generator = self.validation_data_gen.flow_from_directory(
            'dataset/test',
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')
        return validation_data_generator

    def _get_emotion_model_accuracy(self) -> None:
        """Gets the accuracy of model.
        """
        validation_data_generator = self._create_validation_generator()
        emotion_model_accuracy = self.model.evaluate_generator(validation_data_generator)
        emotion_model_accuracy = emotion_model_accuracy[1] * 100
        print(f"Accuracy: {emotion_model_accuracy:.2f}%")

    def start_evaluation(self) -> None:
        """Starts the evaluation process.
        """
        self._load_emotion_model()
        self._compile_emotion_model()
        self._get_emotion_model_accuracy()


if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.start_evaluation()
