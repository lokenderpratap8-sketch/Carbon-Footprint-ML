from flask import Flask, render_template, request
import pickle
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocess import CarbonFootprintPreprocessor
from explainable_ai import ExplainableAI
from recommendation_engine.recommendations import RecommendationEngine

app = Flask(__name__)

# Load ML models and preprocessors
print("Loading ML models...")
try:
    # Load preprocessor
    preprocessor = CarbonFootprintPreprocessor()
    preprocessor.load_preprocessors('models')
    
    # Load regression model
    with open('models/carbon_regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)
    
    # Load classification model
    with open('models/carbon_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    # Load classifier metadata
    with open('models/classifier_metadata.pkl', 'rb') as f:
        classifier_metadata = pickle.load(f)
    
    # Initialize Explainable AI
    xai = ExplainableAI()
    
    # Initialize Recommendation Engine
    recommendation_engine = RecommendationEngine()
    
    print("✓ All ML models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run train_model.py and train_classifier.py first")
    regressor = None
    classifier = None
    preprocessor = None
    xai = None
    recommendation_engine = None

@app.route('/', methods=['GET', 'POST'])
def index():
    total = None
    details = {}
    ml_results = None
    error = None

    if request.method == 'POST':
        try:
            # Transportation Features
            car_km = float(request.form.get('car_km', 0))
            fuel_type = request.form.get('fuel_type', 'petrol')
            public_transport_hours = float(request.form.get('public_transport_hours', 0))
            train_distance = float(request.form.get('train_distance', 0))
            bike_usage = float(request.form.get('bike_usage', 0))
            cab_usage = float(request.form.get('cab_usage', 0))
            vehicle_type = request.form.get('vehicle_type', 'sedan')
            vehicle_age = float(request.form.get('vehicle_age', 0))
            num_vehicles = int(request.form.get('num_vehicles', 1))
            
            # Household Energy Features
            electricity = float(request.form.get('electricity', 0))
            household_members = int(request.form.get('household_members', 1))
            ac_hours = float(request.form.get('ac_hours', 0))
            renewable_energy = int(request.form.get('renewable_energy', 0))
            solar_panels = int(request.form.get('solar_panels', 0))
            lpg_consumption = float(request.form.get('lpg_consumption', 0))
            home_type = request.form.get('home_type', 'apartment')
            
            # Food and Diet Features
            diet_type = request.form.get('diet_type', 'mixed')
            meat_meals = float(request.form.get('meat_meals', 0))
            dairy_consumption = float(request.form.get('dairy_consumption', 0))
            packaged_food = float(request.form.get('packaged_food', 0))
            food_waste = float(request.form.get('food_waste', 0))
            local_food = float(request.form.get('local_food', 0))
            
            # Travel Features
            flight_hours = float(request.form.get('flight_hours', 0))
            domestic_flights = int(request.form.get('domestic_flights', 0))
            international_flights = int(request.form.get('international_flights', 0))
            hotel_stays = int(request.form.get('hotel_stays', 0))
            vacation_frequency = int(request.form.get('vacation_frequency', 0))
            
            # Consumer Behavior Features
            online_shopping = float(request.form.get('online_shopping', 0))
            fast_fashion = int(request.form.get('fast_fashion', 0))
            electronics = int(request.form.get('electronics', 0))
            recycling_score = int(request.form.get('recycling_score', 5))
            waste_segregation = int(request.form.get('waste_segregation', 0))

            # Check if ML models are loaded
            if regressor is None or classifier is None or preprocessor is None:
                error = "ML models not loaded. Please run training scripts first."
                return render_template('index.html', details=details, total=total, ml_results=ml_results, error=error)

            # Prepare input for ML prediction
            input_data = {
                'car_km': car_km,
                'fuel_type': fuel_type,
                'public_transport_hours': public_transport_hours,
                'train_distance': train_distance,
                'bike_usage': bike_usage,
                'cab_usage': cab_usage,
                'vehicle_type': vehicle_type,
                'vehicle_age': vehicle_age,
                'num_vehicles': num_vehicles,
                'electricity': electricity,
                'household_members': household_members,
                'ac_hours': ac_hours,
                'renewable_energy': renewable_energy,
                'solar_panels': solar_panels,
                'lpg_consumption': lpg_consumption,
                'home_type': home_type,
                'diet_type': diet_type,
                'meat_meals': meat_meals,
                'dairy_consumption': dairy_consumption,
                'packaged_food': packaged_food,
                'food_waste': food_waste,
                'local_food': local_food,
                'flight_hours': flight_hours,
                'domestic_flights': domestic_flights,
                'international_flights': international_flights,
                'hotel_stays': hotel_stays,
                'vacation_frequency': vacation_frequency,
                'online_shopping': online_shopping,
                'fast_fashion': fast_fashion,
                'electronics': electronics,
                'recycling_score': recycling_score,
                'waste_segregation': waste_segregation
            }

            # Preprocess input
            try:
                processed_input = preprocessor.preprocess_for_prediction(input_data)
                
                # Predict carbon footprint
                predicted_emission = regressor.predict(processed_input)[0]
                
                # Predict emission category
                category_encoded = classifier.predict(processed_input)[0]
                category = classifier_metadata['reverse_mapping'][category_encoded]
                
                # Get feature importance
                top_contributors = xai.get_top_contributors(processed_input.iloc[0].to_dict())
                
                # Get category breakdown
                category_breakdown = xai.get_category_breakdown(processed_input.iloc[0].to_dict())
                
                # Get recommendations
                recommendations = recommendation_engine.get_personalized_recommendations(category, top_contributors, category_breakdown)
                priority_actions = recommendation_engine.get_priority_actions(category, top_contributors)
                
                # Calculate sustainability score
                sustainability_score = input_data['renewable_energy'] * 20 + input_data['solar_panels'] * 15 + input_data['bike_usage'] * 2 + input_data['recycling_score'] * 5 + input_data['waste_segregation'] * 10 + input_data['local_food'] * 0.1
                
                # ML Results
                ml_results = {
                    "predicted_emission": round(predicted_emission, 2),
                    "emission_category": category,
                    "top_contributors": top_contributors,
                    "category_breakdown": category_breakdown,
                    "recommendations": recommendations,
                    "priority_actions": priority_actions,
                    "model_name": xai.model_name,
                    "sustainability_score": round(sustainability_score, 1)
                }

                total = round(predicted_emission, 2)

            except Exception as e:
                error = f"Error during ML prediction: {str(e)}"
                print(f"Prediction error: {e}")

        except ValueError:
            error = "Please enter valid numbers!"
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', details=details, total=total, ml_results=ml_results, error=error)

if __name__ == "__main__":
    app.run(debug=True)