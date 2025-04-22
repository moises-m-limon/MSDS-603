from metaflow import FlowSpec, step, Parameter, current
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_diabetes


class ModelTrainingFlow(FlowSpec):
    
    seed = Parameter('seed', default=42)
    
    @step
    def start(self):
        """Start the preprocessing."""
        print("\n" + "="*80)
        print("STARTING MODEL TRAINING PIPELINE")
        print(f"Run ID: {current.run_id}")
        print("="*80 + "\n")
        
        # Load diabetes dataset
        data = load_diabetes()
        X = data.data
        y = data.target
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=self.seed
        )
        
        print("Dataset Split Summary:")
        print(f"  • Training samples: {self.X_train.shape[0]:,}")
        print(f"  • Test samples: {self.X_test.shape[0]:,}")
        print(f"  • Features: {self.X_train.shape[1]}\n")
        
        # Store feature names
        self.feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        
        # Fan out to train multiple models in parallel
        self.next(self.train_random_forest, self.train_gradient_boosting)
    
    @step
    def train_random_forest(self):
        """Train Random Forest Regressor."""
        print("\nTraining Random Forest Model...")
        
        # Create and train the model
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions and calculate metrics
        preds = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        
        # Store model and metrics
        self.model_name = "RandomForest"
        self.mse = mse
        self.model = model
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\nRandom Forest Performance Metrics:")
        print(f"  • MSE:  {mse:.4f}")
        print(f"  • RMSE: {rmse:.4f}")
        print(f"  • MAE:  {mae:.4f}")
        print(f"  • R²:   {r2:.4f}\n")
        
        # Continue to the join step
        self.next(self.choose_model)
    
    @step
    def train_gradient_boosting(self):
        """Train Gradient Boosting Regressor."""
        print("\nTraining Gradient Boosting Model...")
        
        # Create and train the model
        model = GradientBoostingRegressor(random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions and calculate metrics
        preds = model.predict(self.X_test)
        
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        
        # Store model and metrics
        self.model_name = "GradientBoosting"
        self.mse = mse
        self.model = model
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\nGradient Boosting Performance Metrics:")
        print(f"  • MSE:  {mse:.4f}")
        print(f"  • RMSE: {rmse:.4f}")
        print(f"  • MAE:  {mae:.4f}")
        print(f"  • R²:   {r2:.4f}\n")
        
        # Continue to the join step
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """Choose best model and register it."""
        print("\n" + "="*80)
        print("MODEL EVALUATION AND SELECTION")
        print("="*80 + "\n")
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment('diabetes-regression')
        
        # Function to evaluate models
        def score(inp):
            preds = inp.model.predict(inp.X_test)
            metrics = {
                'MSE': mean_squared_error(inp.y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(inp.y_test, preds)),
                'MAE': mean_absolute_error(inp.y_test, preds),
                'R2': r2_score(inp.y_test, preds)
            }
            return inp.model, metrics, inp.model_name
        
        # Evaluate all models
        model_results = list(map(score, inputs))
        
        # Sort models by MSE (lower is better)
        sorted_results = sorted(model_results, key=lambda x: x[1]['MSE'])
        
        # Select the best model
        best_model, best_metrics, best_name = sorted_results[0]
        
        # Store results
        self.model = best_model
        self.best_metrics = best_metrics
        self.best_model_name = best_name
        self.results = [(name, metrics) for _, metrics, name in sorted_results]
        
        # Print comparison of all models
        print("Model Comparison:")
        for name, metrics in self.results:
            print(f"\n{name}:")
            print(f"  • MSE:  {metrics['MSE']:.4f}")
            print(f"  • RMSE: {metrics['RMSE']:.4f}")
            print(f"  • MAE:  {metrics['MAE']:.4f}")
            print(f"  • R²:   {metrics['R2']:.4f}")
        
        print(f"\nbest Model: {best_name}")
        
        # Register the best model with MLflow
        print("\nRegistering best model with MLflow...")
        with mlflow.start_run(run_name=f"register_{best_name}"):
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name="diabetes_model"
            )
            
            # Get feature names from first input
            mlflow.log_dict({"features": inputs[0].feature_names}, "feature_names.json")
            
            # Log metrics
            for metric_name, metric_value in best_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        mlflow.end_run()
        print("Model registration complete\n")
        
        # Save test data for scoring demo
        test_data = pd.DataFrame(inputs[0].X_test, columns=inputs[0].feature_names)
        test_data['target'] = inputs[0].y_test
        
        os.makedirs('data/test', exist_ok=True)
        test_data_path = f'data/test/diabetes_test_{current.run_id}.csv'
        test_data.to_csv(test_data_path, index=False)
        print(f"Saved test data to {test_data_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow and summarize results."""
        print("\n" + "="*80)
        print("TRAINING PIPELINE SUMMARY")
        print("="*80 + "\n")
        
        # Print final summary of all models
        print("Final Model Performance:")
        for name, metrics in self.results:
            print(f"\n{name}:")
            print(f"  • MSE:  {metrics['MSE']:.4f}")
            print(f"  • RMSE: {metrics['RMSE']:.4f}")
            print(f"  • MAE:  {metrics['MAE']:.4f}")
            print(f"  • R²:   {metrics['R2']:.4f}")
        
        # Print details of the best model
        print(f"\nbest Model: {self.best_model_name}")
        print(f"best Metrics:")
        print(f"  • MSE:  {self.best_metrics['MSE']:.4f}")
        print(f"  • RMSE: {self.best_metrics['RMSE']:.4f}")
        print(f"  • MAE:  {self.best_metrics['MAE']:.4f}")
        print(f"  • R²:   {self.best_metrics['R2']:.4f}")
        print("\nTraining pipeline completed successfully\n")
        
        # Instructions for scoring
        print("\nTo use this model for scoring, run:")
        print(f"python scoringflow.py run --run_id [mlflow_run_id] --data_path data/test/diabetes_test_{current.run_id}.csv")


if __name__ == '__main__':
    ModelTrainingFlow()