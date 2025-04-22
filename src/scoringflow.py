from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class ModelScoringFlow(FlowSpec):

    seed = Parameter("seed", default=123)  # Use a different seed from training
    
    run_id = Parameter('run_id',
                      help='MLFlow run ID containing the model',
                      default='8b14a22a282041489f86ee59ee1b27c6')

    @step
    def start(self):
        """
        Load new holdout data using the same preprocessing pipeline.
        """
        from sklearn.model_selection import train_test_split

        # Load diabetes dataset
        data = load_diabetes()
        X = data.data
        y = data.target

        # Remove missing targets
        mask = ~pd.isna(y)
        X, y = X[mask], y[mask]

        # Generate a new test set
        _, self.X_test, _, self.y_test = train_test_split(X, y, random_state=self.seed)

        print(f"Holdout test data: {self.X_test.shape[0]} samples")
        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the best model from MLflow.
        """
        mlflow.set_tracking_uri("http://localhost:5001")
        
        if not self.run_id:
            raise ValueError("You must provide an MLFlow run_id parameter")
            
        print(f"Using MLFlow run ID: {self.run_id}")
        
        try:
            model_uri = f"runs:/{self.run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded model from {model_uri}")
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            print(error_msg)
            print(f"Make sure the run ID exists and contains a model artifact")
            raise

        self.next(self.predict)

    @step
    def predict(self):
        """
        Run predictions and compute metrics.
        """
        self.preds = self.model.predict(self.X_test)
        
        # Calculate all metrics
        self.mse = mean_squared_error(self.y_test, self.preds)
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.y_test, self.preds)
        self.r2 = r2_score(self.y_test, self.preds)

        self.next(self.end)

    @step
    def end(self):
        print("\nFinal Scoring Metrics:")
        print(f"  • MSE:  {self.mse:.4f}")
        print(f"  • RMSE: {self.rmse:.4f}")
        print(f"  • MAE:  {self.mae:.4f}")
        print(f"  • R²:   {self.r2:.4f}")
        
        print("\nSample Predictions:")
        print(pd.DataFrame({
            "y_true": self.y_test[:5],
            "y_pred": self.preds[:5]
        }))


if __name__ == "__main__":
    ModelScoringFlow()