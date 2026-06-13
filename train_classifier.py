import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing.preprocess_v2 import CarbonFootprintPreprocessorV2

class CarbonFootprintClassifierV2:
    def __init__(self):
        self.classifier = None
        self.category_mapping = {
            'Low': 0,
            'Medium': 1,
            'High': 2,
            'Critical': 3
        }
        self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
        
    def create_emission_categories(self, df):
        """Create emission categories based on annual carbon emission"""
        df = df.copy()
        
        # Define thresholds based on dataset percentiles
        percentiles = df['annual_carbon_emission'].quantile([0.25, 0.5, 0.75])
        
        conditions = [
            df['annual_carbon_emission'] <= percentiles[0.25],
            (df['annual_carbon_emission'] > percentiles[0.25]) & (df['annual_carbon_emission'] <= percentiles[0.5]),
            (df['annual_carbon_emission'] > percentiles[0.5]) & (df['annual_carbon_emission'] <= percentiles[0.75]),
            df['annual_carbon_emission'] > percentiles[0.75]
        ]
        
        categories = ['Low', 'Medium', 'High', 'Critical']
        
        df['emission_category'] = np.select(conditions, categories, default='Medium')
        
        print(f"Emission Category Thresholds:")
        print(f"  Low: ≤ {percentiles[0.25]:.2f} kg CO₂")
        print(f"  Medium: {percentiles[0.25]:.2f} - {percentiles[0.5]:.2f} kg CO₂")
        print(f"  High: {percentiles[0.5]:.2f} - {percentiles[0.75]:.2f} kg CO₂")
        print(f"  Critical: > {percentiles[0.75]:.2f} kg CO₂")
        print(f"\nCategory Distribution:")
        print(df['emission_category'].value_counts())
        
        return df
    
    def train_classifier(self, X_train, y_train):
        """Train Random Forest Classifier"""
        print("\nTraining Random Forest Classifier...")
        
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        print("✓ Classifier trained")
        
        return self.classifier
    
    def evaluate_classifier(self, X_test, y_test):
        """Evaluate classifier performance"""
        y_pred = self.classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("CLASSIFIER PERFORMANCE")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_classifier(self, output_dir='models'):
        """Save the classifier"""
        os.makedirs(output_dir, exist_ok=True)
        
        classifier_path = f'{output_dir}/carbon_classifier_v2.pkl'
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"\n✓ Classifier saved to {classifier_path}")
        
        # Save category mapping
        metadata = {
            'category_mapping': self.category_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        
        metadata_path = f'{output_dir}/classifier_metadata_v2.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Classifier metadata saved to {metadata_path}")

def main():
    print("="*60)
    print("CARBON FOOTPRINT CLASSIFIER TRAINING V2")
    print("Comprehensive Feature Set")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('dataset/carbon_footprint_dataset_v2.csv')
    print(f"✓ Dataset loaded: {len(df)} samples")
    
    # Create emission categories
    print("\nCreating emission categories...")
    classifier = CarbonFootprintClassifierV2()
    df = classifier.create_emission_categories(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = CarbonFootprintPreprocessorV2()
    
    # Get emission categories before preprocessing
    y_categories = df['emission_category']
    
    # Preprocess features
    X_train, X_test, y_train_emission, y_test_emission, feature_cols = preprocessor.preprocess_for_training(df)
    
    # Get the indices from the train/test split
    train_indices = X_train.index
    test_indices = X_test.index
    
    # Convert categories to numerical labels using the indices
    y_train = y_categories.iloc[train_indices].map(classifier.category_mapping)
    y_test = y_categories.iloc[test_indices].map(classifier.category_mapping)
    
    print(f"✓ Data preprocessed and labeled")
    
    # Train classifier
    classifier.train_classifier(X_train, y_train)
    
    # Evaluate classifier
    classifier.evaluate_classifier(X_test, y_test)
    
    # Save classifier
    classifier.save_classifier()
    
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
