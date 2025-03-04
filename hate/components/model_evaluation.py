import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import MAX_LEN
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self,
                 evaluation_config: ModelEvaluationConfig,
                 trainer_artifacts: ModelTrainerArtifacts,
                 transformation_artifacts: DataTransformationArtifacts):
        """
        Initializes the ModelEvaluation class with required configuration and artifact information.
        :param evaluation_config: Configuration for model evaluation.
        :param trainer_artifacts: Artifacts produced by the model training stage.
        :param transformation_artifacts: Artifacts produced by the data transformation stage.
        """
        self.evaluation_config = evaluation_config
        self.trainer_artifacts = trainer_artifacts
        self.transformation_artifacts = transformation_artifacts

    def fetch_best_model_path(self) -> str:
        """
        Creates the best model directory (if not exists) and returns the path for the best model file.
        :return: Full path of the best model.
        """
        try:
            os.makedirs(self.evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            best_model_path = os.path.join(self.evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.evaluation_config.MODEL_NAME)
            logging.info("Fetched best model path from cloud storage")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_test_data(self):
        """
        Loads the test data and prepares the text sequences.
        :return: A tuple of (padded_test_sequences, y_test, tokenizer)
        """
        try:
            # Load test data
            x_test = pd.read_csv(self.trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.trainer_artifacts.y_test_path, index_col=0)

            # Load tokenizer from pickle file
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # Convert tweet column to string and prepare sequences
            x_test = x_test['tweet'].astype(str).squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            padded_sequences = pad_sequences(test_sequences, maxlen=MAX_LEN)

            return padded_sequences, y_test, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self, model) -> float:
        """
        Evaluates the given model on the test dataset.
        :param model: Keras model to evaluate.
        :return: Evaluation metric (accuracy or loss depending on model.compile) of the model.
        """
        try:
            padded_sequences, y_test, _ = self._load_test_data()

            # Evaluate the model on test data
            evaluation_score = model.evaluate(padded_sequences, y_test, verbose=0)
            logging.info(f"Evaluation score: {evaluation_score}")

            # Get predictions and compute confusion matrix
            predictions = model.predict(padded_sequences)
            predicted_labels = [0 if pred[0] < 0.5 else 1 for pred in predictions]
            conf_matrix = confusion_matrix(y_test, predicted_labels)
            logging.info(f"Confusion Matrix: {conf_matrix}")

            return evaluation_score
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Initiates the model evaluation process:
         - Loads the currently trained model and evaluates it.
         - Fetches the best model (if available) from storage and evaluates it.
         - Compares the evaluation metrics and decides whether to accept the new model.
        :return: ModelEvaluationArtifacts with the decision.
        """
        logging.info("Initiating model evaluation process")
        try:
            # Load the currently trained model and evaluate it
            trained_model = keras.models.load_model(self.trainer_artifacts.trained_model_path)
            trained_model_score = self.evaluate_model(trained_model)

            # Fetch best model path from cloud storage
            best_model_path = self.fetch_best_model_path()

            # If no best model exists, accept the trained model
            if not os.path.isfile(best_model_path):
                accept_new_model = True
                logging.info("No best model found. Accepting the newly trained model.")
            else:
                # Load and evaluate the best model
                best_model = keras.models.load_model(best_model_path)
                best_model_score = self.evaluate_model(best_model)

                # Accept the new model only if it outperforms the best model
                if trained_model_score > best_model_score:
                    accept_new_model = True
                    logging.info("Newly trained model outperforms the best model. New model accepted.")
                else:
                    accept_new_model = False
                    logging.info("Newly trained model did not outperform the best model. New model rejected.")

            evaluation_artifact = ModelEvaluationArtifacts(is_model_accepted=accept_new_model)
            logging.info("Model evaluation process completed")
            return evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
