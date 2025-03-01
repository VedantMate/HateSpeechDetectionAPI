import os
import sys
import keras
import pickle
from PIL import Image
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class PredictionPipeline:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.data_transformation = DataTransformation(
            data_transformation_config=DataTransformationConfig,
            data_ingestion_artifacts=DataIngestionArtifacts
        )

    def get_model_from_local(self) -> str:
        """
        Method Name :   get_model_from_local
        Description :   This method loads the model from local directory
        Output      :   best_model_path
        """
        logging.info("Entered the get_model_from_local method of PredictionPipeline class")
        try:
            # Ensure the model directory exists
            os.makedirs(self.model_path, exist_ok=True)
            best_model_path = os.path.join(self.model_path, self.model_name)

            # Check if the model file exists locally
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Model {self.model_name} not found locally.")

            logging.info(f"Model loaded from {best_model_path}")
            logging.info("Exited the get_model_from_local method of PredictionPipeline class")

            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, best_model_path, text):
        """load model, preprocess text, and make predictions"""
        logging.info("Running the predict function")
        try:
            # Load the model
            load_model = keras.models.load_model(best_model_path)

            # Load the tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            # Data transformation and cleaning
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]  # Model expects a list of text
            
            # Text preprocessing
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)

            # Prediction
            pred = load_model.predict(padded)

            # Return prediction result
            if pred > 0.5:
                logging.info("Predicted: hate and abusive")
                return "hate and abusive"
            else:
                logging.info("Predicted: no hate")
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            # Load model from local directory
            best_model_path = self.get_model_from_local()

            # Run prediction on the provided text
            predicted_text = self.predict(best_model_path, text)

            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text

        except Exception as e:
            raise CustomException(e, sys) from e
