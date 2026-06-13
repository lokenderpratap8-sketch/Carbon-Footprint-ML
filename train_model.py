import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from preprocessing.preprocess_v2 import CarbonFootprintPreprocessorV2

class CarbonFootprintModelTrainerV2:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        print("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        print("✓ Linear Regression trained")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest Regressor"""
        print("Training Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("✓ Random Forest trained")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost Regressor"""
        print("Training XGBoost Regressor...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        print("✓ XGBoost trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        return {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def select_best_model(self, X_test, y_test):
        """Select best model based on R² score"""
        print("\n" + "="*50)
        print("SELECTING BEST MODEL")
        print("="*50)
        
        results = []
        for name, model in self.models.items():
            result = self.evaluate_model(model, X_test, y_test, name)
            results.append(result)
        
        # Select model with highest R² score
        best_result = max(results, key=lambda x: x['r2'])
        self.best_model = self.models[best_result['model_name']]
        self.best_model_name = best_result['model_name']
        
        print("\n" + "="*50)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"R² Score: {best_result['r2']:.4f}")
        print("="*50)
        
        return best_result, results
    
    def save_best_model(self, output_dir='models'):
        """Save the best model"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = f'{output_dir}/carbon_regressor_v2.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"\n✓ Best model saved to {model_path}")
        
        # Save model name for reference
        metadata = {
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns)
        }
        
        metadata_path = f'{output_dir}/model_metadata_v2.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Model metadata saved to {metadata_path}")

def main():
    print("="*60)
    print("CARBON FOOTPRINT MODEL TRAINING V2")
    print("Comprehensive Feature Set (40 Features)")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('dataset/carbon_footprint_dataset_v2.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = CarbonFootprintPreprocessorV2()
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.preprocess_for_training(df)
    print(f"✓ Data preprocessed")
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    # Initialize trainer
    trainer = CarbonFootprintModelTrainerV2()
    trainer.feature_columns = feature_cols
    
    # Train all models
    trainer.train_linear_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_xgboost(X_train, y_train)
    
    # Select best model
    best_result, all_results = trainer.select_best_model(X_test, y_test)
    
    # Save best model
    trainer.save_best_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
