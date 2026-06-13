import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class CarbonFootprintPreprocessorV2:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Categorical columns
        self.categorical_columns = [
            'fuel_type',
            'vehicle_type',
            'home_type',
            'diet_type'
        ]
        
        # Numerical columns
        self.numerical_columns = [
            'car_km',
            'public_transport_hours',
            'train_distance',
            'bike_usage',
            'cab_usage',
            'vehicle_age',
            'num_vehicles',
            'electricity',
            'household_members',
            'ac_hours',
            'renewable_energy',
            'solar_panels',
            'lpg_consumption',
            'meat_meals',
            'dairy_consumption',
            'packaged_food',
            'food_waste',
            'local_food',
            'flight_hours',
            'domestic_flights',
            'international_flights',
            'hotel_stays',
            'vacation_frequency',
            'online_shopping',
            'fast_fashion',
            'electronics',
            'recycling_score',
            'waste_segregation'
        ]
        
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            if fit:
                encoder = LabelEncoder()
                df[col + '_encoded'] = encoder.fit_transform(df[col])
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders[col]
                # Handle unseen categories
                df[col + '_encoded'] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        df = df.copy()
        
        # Combine numerical and encoded categorical columns
        columns_to_scale = self.numerical_columns + [col + '_encoded' for col in self.categorical_columns]
        
        if fit:
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
        else:
            df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        return df
    
    def feature_engineering(self, df):
        """Create derived features"""
        df = df.copy()
        
        # 1. Annual Travel Distance (km/year)
        df['annual_travel_distance'] = (
            df['car_km'] * 12 +
            df['train_distance'] +
            df['cab_usage'] * 12 * 10  # Assume 10km per cab trip
        )
        
        # 2. Annual Energy Consumption (kWh/year)
        df['annual_energy_consumption'] = (
            df['electricity'] * 12 +
            df['ac_hours'] * 365 * 1.5  # AC in kWh equivalent
        )
        
        # 3. Household Carbon Intensity (kg CO2 per person)
        df['household_carbon_intensity'] = df['annual_energy_consumption'] / df['household_members']
        
        # 4. Travel Intensity Score (normalized)
        df['travel_intensity_score'] = (
            df['car_km'] +
            df['public_transport_hours'] * 10 +
            df['train_distance'] * 0.1 +
            df['flight_hours'] * 50
        ) / 1000  # Normalize to reasonable scale
        
        # 5. Sustainable Lifestyle Score (higher = more sustainable)
        df['sustainability_score'] = (
            df['renewable_energy'] * 20 +
            df['solar_panels'] * 15 +
            df['bike_usage'] * 2 +
            df['recycling_score'] * 5 +
            df['waste_segregation'] * 10 +
            df['local_food'] * 0.1
        )
        
        # 6. Diet Impact Score
        df['diet_impact_score'] = (
            df['meat_meals'] * 7 +
            df['dairy_consumption'] * 1.5 -
            df['local_food'] * 0.05
        )
        
        # 7. Consumer Impact Score
        df['consumer_impact_score'] = (
            df['online_shopping'] * 0.5 +
            df['fast_fashion'] * 10 +
            df['electronics'] * 50 -
            df['recycling_score'] * 10
        )
        
        # 8. Flight Intensity
        df['flight_intensity'] = (
            df['domestic_flights'] * 200 +
            df['international_flights'] * 800
        )
        
        return df
    
    def preprocess_for_training(self, df):
        """Complete preprocessing pipeline for training"""
        print("Starting preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        print("✓ Categorical features encoded")
        
        # Feature engineering
        df = self.feature_engineering(df)
        print("✓ Feature engineering completed")
        
        # Prepare feature columns
        base_features = self.numerical_columns + [col + '_encoded' for col in self.categorical_columns]
        engineered_features = [
            'annual_travel_distance',
            'annual_energy_consumption',
            'household_carbon_intensity',
            'travel_intensity_score',
            'sustainability_score',
            'diet_impact_score',
            'consumer_impact_score',
            'flight_intensity'
        ]
        
        all_features = base_features + engineered_features
        
        # Scale features
        df[all_features] = self.scaler.fit_transform(df[all_features])
        print("✓ Features scaled")
        
        # Prepare X and y
        X = df[all_features]
        y = df['annual_carbon_emission']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test, all_features
    
    def preprocess_for_prediction(self, input_data):
        """Preprocess single input for prediction"""
        df = pd.DataFrame([input_data])
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Prepare feature columns
        base_features = self.numerical_columns + [col + '_encoded' for col in self.categorical_columns]
        engineered_features = [
            'annual_travel_distance',
            'annual_energy_consumption',
            'household_carbon_intensity',
            'travel_intensity_score',
            'sustainability_score',
            'diet_impact_score',
            'consumer_impact_score',
            'flight_intensity'
        ]
        
        all_features = base_features + engineered_features
        
        # Scale features
        df[all_features] = self.scaler.transform(df[all_features])
        
        return df[all_features]
    
    def save_preprocessors(self, output_dir='models'):
        """Save encoders and scaler"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/encoders_v2.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(f'{output_dir}/scaler_v2.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Preprocessors saved to {output_dir}")
    
    def load_preprocessors(self, input_dir='models'):
        """Load encoders and scaler"""
        with open(f'{input_dir}/encoders_v2.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f'{input_dir}/scaler_v2.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Preprocessors loaded from {input_dir}")

if __name__ == "__main__":
    print("Testing preprocessing pipeline v2...")
    
    df = pd.read_csv('dataset/carbon_footprint_dataset_v2.csv')
    print(f"Dataset loaded: {len(df)} samples")
    
    preprocessor = CarbonFootprintPreprocessorV2()
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.preprocess_for_training(df)
    
    preprocessor.save_preprocessors()
    
    print("\nPreprocessing completed successfully!")
    print(f"Total features: {len(feature_cols)}")
    print(f"Base features: {len(preprocessor.numerical_columns) + len(preprocessor.categorical_columns)}")
    print(f"Engineered features: 8")
