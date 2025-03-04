import os
import sys
import keras
import pickle
from keras.utils import pad_sequences
from hate.logger import logging
from hate.constants import MODEL_NAME  
from hate.exception import CustomException
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

# Define the paths for your model and tokenizer
MODEL_PATH = os.path.join("artifacts", "PredictModel", MODEL_NAME)
TOKENIZER_PATH = "tokenizer.pickle"  

try:
    GLOBAL_MODEL = keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        GLOBAL_TOKENIZER = pickle.load(handle)
except Exception as e:
    raise CustomException(e, sys)

# -----------------------------------------------
# PredictionPipeline Class Definition
# -----------------------------------------------

class PredictionPipeline:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        # Create an instance of DataTransformation.
        # IMPORTANT: Pass in actual configuration instances instead of the classes themselves.
        self.data_transformation = DataTransformation(
            data_transformation_config=DataTransformationConfig,  # Provide a configured instance if available
            data_ingestion_artifacts=DataIngestionArtifacts        # Provide a configured instance if available
        )

    def predict(self, text: str) -> str:
        """
        Load preprocessed text, convert to sequences using the global tokenizer, 
        pad sequences, and make a prediction using the global model.
        """
        #logging.info("Running the predict function")
        try:
            # Clean and transform the input text
            transformed_text = self.data_transformation.concat_data_cleaning(text)
            # The tokenizer expects a list of texts
            text_list = [transformed_text]
            
            # Tokenize using the preloaded global tokenizer
            sequences = GLOBAL_TOKENIZER.texts_to_sequences(text_list)
            # Pad the sequences (ensure maxlen is set appropriately; here it is hardcoded to 300)
            padded = pad_sequences(sequences, maxlen=300)
            
            # Use the preloaded global model to predict
            pred = GLOBAL_MODEL.predict(padded)
            # Extract the prediction score (assuming a single output value)
            prediction_score = pred[0][0] if pred.ndim > 1 else pred[0]
            #logging.info(f"Prediction score: {prediction_score}")
            
            # Return a label based on the prediction threshold
            if prediction_score > 0.5:
                #logging.info("Predicted: hate and abusive")
                return "hate and abusive"
            else:
                #logging.info("Predicted: no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text: str) -> str:
        """
        Entry point to run the prediction pipeline.
        """
        #logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            result = self.predict(text)
            #logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return result
        except Exception as e:
            raise CustomException(e, sys) from e

# -----------------------------------------------
# Example Usage:
# In your API endpoint, you can initialize PredictionPipeline once 
# and then call run_pipeline() for each request.
#
# Example (e.g., using Flask):
#
# from flask import Flask, request, jsonify
# app = Flask(__name__)
# pipeline = PredictionPipeline()  # Initialize once on startup
#
# @app.route('/predict', methods=['POST'])
# def predict_endpoint():
#     data = request.get_json()
#     text = data.get("text", "")
#     try:
#         result = pipeline.run_pipeline(text)
#         return jsonify({"prediction": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run()
# -----------------------------------------------

