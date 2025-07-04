import os
import joblib
import yaml
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
import mlflow
import mlflow.sklearn

# Read params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Dagshub MLflow config
dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")

if dagshub_username and dagshub_token:
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_username}/cancer.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    print(f"MLflow tracking URI set to: https://dagshub.com/{dagshub_username}/cancer.mlflow")
else:
    print("Warning: DagHub credentials not found. Using local MLflow tracking.")

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Load params from YAML
        self.n_estimators = params["n_estimators"]
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]

        logger.info("Model Training Initialized.")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            logger.info("Processed data loaded for model training.")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise CustomException("Failed to load data for training")

    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))

            mlflow.sklearn.log_model(self.model, artifact_path="model")

            # Log parameters
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth
            })

            logger.info("Model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Model training failed")

    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1] if len(self.y_test.unique()) == 2 else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else None

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            # Save metrics to file
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc if roc_auc is not None else "N/A"
            }
            with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Model evaluation complete. Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Model evaluation failed")

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()
