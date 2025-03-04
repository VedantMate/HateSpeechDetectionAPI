import os
import sys
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with configuration details.
        :param data_ingestion_config: Configuration for data ingestion.
        """
        self.data_ingestion_config = data_ingestion_config
        # Define the ZIP file path where the dataset is stored locally.
        self.ZIP_FILE_PATH = os.path.join(os.getcwd(), "data", "dataset.zip")  

    def get_data_locally(self) -> None:
        """
        Checks if the dataset ZIP file exists locally.
        Raises an exception if the file is missing.
        """
        try:
            logging.info("Entered the get_data_locally method of Data ingestion class")
            
            # Check if the ZIP file exists
            if not os.path.exists(self.ZIP_FILE_PATH):
                raise FileNotFoundError(f"ZIP file {self.ZIP_FILE_PATH} not found locally.")

            # Ensure that the required directory for storing artifacts exists
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            logging.info(f"Found ZIP file locally at {self.ZIP_FILE_PATH}")
            logging.info("Exited the get_data_locally method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        """
        Unzips the dataset and returns the paths to extracted files.
        """
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            # Open and extract the ZIP file into the specified directory
            with ZipFile(self.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            # Return extracted data paths
            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Main function that handles the entire data ingestion process.
        """
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            # Step 1: Fetch the dataset locally
            self.get_data_locally()
            logging.info("Fetched the data locally")

            # Step 2: Unzip and clean the dataset
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and split into train and valid datasets")

            # Step 3: Create a DataIngestionArtifacts object with extracted file paths
            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e