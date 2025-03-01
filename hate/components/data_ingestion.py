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
        Initializes the DataIngestion class with the provided configuration.
        :param data_ingestion_config: Configuration for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config
        # Define the ZIP file path inside the constructor
        self.ZIP_FILE_PATH = os.path.join(os.getcwd(), "data", "dataset.zip")  # Local path for the zip file

    def get_data_locally(self) -> None:
        """
        This method handles local file fetching instead of Google Cloud.
        """
        try:
            logging.info("Entered the get_data_locally method of Data ingestion class")
            # Check if the ZIP file exists locally
            if not os.path.exists(self.ZIP_FILE_PATH):
                raise FileNotFoundError(f"ZIP file {self.ZIP_FILE_PATH} not found locally.")

            # No need to sync from GCP, just ensure the directory is created
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            logging.info(f"Found ZIP file locally at {self.ZIP_FILE_PATH}")
            logging.info("Exited the get_data_locally method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        """
        Unzips the data and cleans up, returning the paths to the required files.
        """
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            # Unzips the data to the specified directory
            with ZipFile(self.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        The main method to initiate data ingestion, which fetches, unzips, and prepares the data.
        """
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            # Fetch data locally instead of GCP
            self.get_data_locally()
            logging.info("Fetched the data locally")

            # Unzip and clean the data
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and split into train and valid")

            # Create the DataIngestionArtifacts with paths
            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )



            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

