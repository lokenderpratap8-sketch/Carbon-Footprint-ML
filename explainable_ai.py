import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

class ExplainableAIV2:
    def __init__(self, model_path='models/carbon_regressor_v2.pkl', 
                 scaler_path='models/scaler_v2.pkl',
                 metadata_path='models/model_metadata_v2.pkl',
                 encoders_path='models/encoders_v2.pkl'):
        """Initialize Explainable AI with loaded model and preprocessors"""
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.feature_columns = self.metadata['feature_columns']
        self.model_name = self.metadata['model_name']
        
        # Feature names for display (grouped by category)
        self.feature_display_names = {
            # Transportation
            'car_km': 'Car Travel (km/month)',
            'fuel_type_encoded': 'Fuel Type',
            'public_transport_hours': 'Public Transport (hrs/week)',
            'train_distance': 'Train Travel (km/year)',
            'bike_usage': 'Bike Usage (hrs/week)',
            'cab_usage': 'Cab/Rideshare (trips/month)',
            'vehicle_type_encoded': 'Vehicle Type',
            'vehicle_age': 'Vehicle Age (years)',
            'num_vehicles': 'Number of Vehicles',
            
            # Household Energy
            'electricity': 'Electricity (kWh/month)',
            'household_members': 'Household Members',
            'ac_hours': 'AC Usage (hrs/day)',
            'renewable_energy': 'Renewable Energy',
            'solar_panels': 'Solar Panels',
            'lpg_consumption': 'LPG Consumption (kg/month)',
            'home_type_encoded': 'Home Type',
            
            # Food and Diet
            'diet_type_encoded': 'Diet Type',
            'meat_meals': 'Meat Meals (per week)',
            'dairy_consumption': 'Dairy Consumption (servings/day)',
            'packaged_food': 'Packaged Food (meals/week)',
            'food_waste': 'Food Waste (%)',
            'local_food': 'Local Food Consumption (%)',
            
            # Travel
            'flight_hours': 'Flight Hours (per year)',
            'domestic_flights': 'Domestic Flights (per year)',
            'international_flights': 'International Flights (per year)',
            'hotel_stays': 'Hotel Stays (per year)',
            'vacation_frequency': 'Vacation Frequency (trips/year)',
            
            # Consumer Behavior
            'online_shopping': 'Online Shopping (USD/month)',
            'fast_fashion': 'Fast Fashion (items/month)',
            'electronics': 'Electronics (devices/year)',
            'recycling_score': 'Recycling Score (1-10)',
            'waste_segregation': 'Waste Segregation',
            
            # Engineered Features
            'annual_travel_distance': 'Annual Travel Distance',
            'annual_energy_consumption': 'Annual Energy Consumption',
            'household_carbon_intensity': 'Household Carbon Intensity',
            'travel_intensity_score': 'Travel Intensity Score',
            'sustainability_score': 'Sustainability Score',
            'diet_impact_score': 'Diet Impact Score',
            'consumer_impact_score': 'Consumer Impact Score',
            'flight_intensity': 'Flight Intensity'
        }
        
        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Features: {len(self.feature_columns)}")
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        
        if self.model_name == 'Random Forest':
            importances = self.model.feature_importances_
        elif self.model_name == 'XGBoost':
            importances = self.model.feature_importances_
        elif self.model_name == 'Linear Regression':
            importances = np.abs(self.model.coef_)
        else:
            importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        
        # Normalize to percentages
        importances = 100 * (importances / importances.sum())
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': [self.feature_display_names.get(f, f) for f in self.feature_columns],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_shap_explanation(self, input_data):
        """Get SHAP values for a single prediction"""
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get feature importance for the prediction
        if self.model_name == 'XGBoost':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(df)
            
            # Get absolute SHAP values
            abs_shap = np.abs(shap_values[0])
            
            # Normalize to percentages
            shap_percentages = 100 * (abs_shap / abs_shap.sum())
            
            # Create SHAP explanation dataframe
            shap_explanation = pd.DataFrame({
                'feature': [self.feature_display_names.get(f, f) for f in self.feature_columns],
                'contribution': shap_percentages
            }).sort_values('contribution', ascending=False)
            
            return shap_explanation
        else:
            # For other models, use feature importance as approximation
            return self.get_feature_importance()
    
    def get_top_contributors(self, input_data, top_n=5):
        """Get top N contributors to the prediction"""
        
        shap_explanation = self.get_shap_explanation(input_data)
        
        # Get top N contributors
        top_contributors = shap_explanation.head(top_n)
        
        # Format as dictionary
        contributors = {}
        for _, row in top_contributors.iterrows():
            contributors[row['feature']] = round(row['contribution'], 1)
        
        return contributors
    
    def get_category_breakdown(self, input_data):
        """Get carbon footprint breakdown by category"""
        
        df = pd.DataFrame([input_data])
        
        # Calculate approximate contributions by category
        breakdown = {
            'Transportation': 0,
            'Household Energy': 0,
            'Food and Diet': 0,
            'Travel': 0,
            'Consumer Behavior': 0
        }
        
        # Transportation features
        transport_features = ['car_km', 'public_transport_hours', 'train_distance', 
                           'bike_usage', 'cab_usage', 'vehicle_age', 'num_vehicles']
        for feat in transport_features:
            if feat in df.columns:
                breakdown['Transportation'] += df[feat].values[0] * 10  # Approximate multiplier
        
        # Household Energy features
        energy_features = ['electricity', 'ac_hours', 'lpg_consumption', 'household_members']
        for feat in energy_features:
            if feat in df.columns:
                breakdown['Household Energy'] += df[feat].values[0] * 8
        
        # Food and Diet features
        food_features = ['meat_meals', 'dairy_consumption', 'packaged_food', 'food_waste']
        for feat in food_features:
            if feat in df.columns:
                breakdown['Food and Diet'] += df[feat].values[0] * 5
        
        # Travel features
        travel_features = ['flight_hours', 'domestic_flights', 'international_flights', 
                         'hotel_stays', 'vacation_frequency']
        for feat in travel_features:
            if feat in df.columns:
                breakdown['Travel'] += df[feat].values[0] * 15
        
        # Consumer Behavior features
        consumer_features = ['online_shopping', 'fast_fashion', 'electronics']
        for feat in consumer_features:
            if feat in df.columns:
                breakdown['Consumer Behavior'] += df[feat].values[0] * 3
        
        # Normalize to percentages
        total = sum(breakdown.values())
        if total > 0:
            breakdown = {k: round(100 * v / total, 1) for k, v in breakdown.items()}
        
        return breakdown
    
    def get_explanation_summary(self, input_data):
        """Get complete explanation summary for a prediction"""
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Get top contributors for this specific prediction
        top_contributors = self.get_top_contributors(input_data)
        
        # Get category breakdown
        category_breakdown = self.get_category_breakdown(input_data)
        
        summary = {
            'model_name': self.model_name,
            'global_feature_importance': feature_importance.head(10).to_dict('records'),
            'top_contributors': top_contributors,
            'category_breakdown': category_breakdown
        }
        
        return summary

if __name__ == "__main__":
    print("Testing Explainable AI v2...")
    
    # Sample input with all features
    sample_input = {
        'car_km': 400,
        'fuel_type_encoded': 0,
        'public_transport_hours': 5,
        'train_distance': 1000,
        'bike_usage': 3,
        'cab_usage': 5,
        'vehicle_type_encoded': 1,
        'vehicle_age': 5,
        'num_vehicles': 2,
        'electricity': 450,
        'household_members': 3,
        'ac_hours': 4,
        'renewable_energy': 0,
        'solar_panels': 0,
        'lpg_consumption': 20,
        'home_type_encoded': 0,
        'diet_type_encoded': 0,
        'meat_meals': 8,
        'dairy_consumption': 2,
        'packaged_food': 5,
        'food_waste': 10,
        'local_food': 40,
        'flight_hours': 10,
        'domestic_flights': 2,
        'international_flights': 1,
        'hotel_stays': 3,
        'vacation_frequency': 2,
        'online_shopping': 200,
        'fast_fashion': 2,
        'electronics': 1,
        'recycling_score': 6,
        'waste_segregation': 1,
        'annual_travel_distance': 6000,
        'annual_energy_consumption': 6000,
        'household_carbon_intensity': 2000,
        'travel_intensity_score': 2,
        'sustainability_score': 30,
        'diet_impact_score': 50,
        'consumer_impact_score': 100,
        'flight_intensity': 1200
    }
    
    # Initialize Explainable AI
    xai = ExplainableAIV2()
    
    # Get feature importance
    print("\n" + "="*50)
    print("GLOBAL FEATURE IMPORTANCE (TOP 10)")
    print("="*50)
    feature_importance = xai.get_feature_importance()
    print(feature_importance.head(10).to_string(index=False))
    
    # Get SHAP explanation
    print("\n" + "="*50)
    print("TOP CONTRIBUTORS FOR PREDICTION")
    print("="*50)
    top_contributors = xai.get_top_contributors(sample_input)
    for feature, contribution in top_contributors.items():
        print(f"{feature}: {contribution}%")
    
    # Get category breakdown
    print("\n" + "="*50)
    print("CATEGORY BREAKDOWN")
    print("="*50)
    category_breakdown = xai.get_category_breakdown(sample_input)
    for category, percentage in category_breakdown.items():
        print(f"{category}: {percentage}%")
    
    # Get complete summary
    print("\n" + "="*50)
    print("EXPLANATION SUMMARY")
    print("="*50)
    summary = xai.get_explanation_summary(sample_input)
    print(f"Model: {summary['model_name']}")
    print(f"Top Contributors: {summary['top_contributors']}")
    print(f"Category Breakdown: {summary['category_breakdown']}")
    
    print("\n✓ Explainable AI v2 test completed successfully!")
