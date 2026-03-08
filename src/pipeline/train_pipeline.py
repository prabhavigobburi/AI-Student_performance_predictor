from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def run_pipeline(self):

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transform = DataTransformation()
        X_train, X_test, y_train, y_test = transform.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        trainer.initiate_model_trainer(X_train, y_train)


if __name__ == "__main__":
    
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

    print("Training completed. Files saved in artifacts folder.")