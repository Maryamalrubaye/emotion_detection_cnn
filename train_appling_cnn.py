import cv2

from keras import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, Flatten, BatchNormalization
)


class EmotionClassifier:
    def __init__(self, train_data_dir: dict, validation_data_dir: dict):
        self.train_data_gen = ImageDataGenerator(rescale=1. / 255)
        self.validation_data_gen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = self.train_data_gen.flow_from_directory(
            train_data_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

        self.validation_generator = self.validation_data_gen.flow_from_directory(
            validation_data_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

        self.model = EmotionClassifier._build_emotion_model()

    @staticmethod
    def _build_emotion_model() -> Sequential:
        """Build CNN model

        Returns:
            The compiled CNN model
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-5), metrics=['accuracy'])

        return model

    def train_emotion_model(self, epochs: int) -> None:
        """Train the CNN model on the training data for the specified number of epochs

        Args:
            epochs: The number of epochs to train the model
        """
        cv2.ocl.setUseOpenCL(False)
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=28709 // 64,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=7178 // 64)

    def save_emotion_model(self, model_json_file: str, model_weights_file: str) -> None:
        """Save the architecture and weights of the CNN model to the specified files

        Args:
            model_json_file: file to save the model architecture
            model_weights_file: file to save the model weights
        """
        model_json = self.model.to_json()
        with open(model_json_file, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights_file)


if __name__ == '__main__':
    classifier = EmotionClassifier('dataset/train', 'dataset/test')
    classifier.train_emotion_model(60)
    classifier.save_emotion_model('model/emotion_detection_model.json', 'model/emotion_detection_model.h5')
