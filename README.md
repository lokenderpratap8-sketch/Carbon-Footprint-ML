# Carbon Footprint ML (EcoTrack)
AI-Based Carbon Footprint Prediction and Recommendation System.
An advanced Machine Learning-powered carbon footprint calculator that provides personalized predictions, emission categorization, explainable AI insights, and actionable recommendations across 31 lifestyle features.

## 🌍 Project Overview

This project transforms a traditional formula-based carbon footprint calculator into an intelligent AI/ML system that:

- **Predicts** annual carbon emissions using trained regression models (XGBoost)
- **Classifies** emissions into categories (Low, Medium, High, Critical)
- **Explains** predictions using SHAP values and feature importance
- **Recommends** personalized actions to reduce carbon footprint
- **Deploys** via modern Flask web application with Tabler Icons

### Feature Categories (31 Base Features)

The system analyzes carbon footprint across five comprehensive categories:

1. **Transportation** (8 features): Car travel, fuel type, public transport, train travel, bike usage, cab/rideshare, vehicle type, vehicle age, number of vehicles
2. **Household Energy** (7 features): Electricity consumption, household members, AC usage, LPG consumption, renewable energy, solar panels, home type
3. **Food & Diet** (6 features): Diet type, meat meals, dairy consumption, packaged food, food waste, local food consumption
4. **Travel** (5 features): Flight hours, domestic flights, international flights, hotel stays, vacation frequency
5. **Consumer Behavior** (5 features): Online shopping, fast fashion, electronics, recycling score, waste segregation

## 🏗️ Project Structure

```
carbon Footprint/
│
├── app.py                          # Flask application with ML integration
├── train_model.py                  # Regression model training script
├── train_classifier.py             # Classification model training script
├── generate_dataset.py             # Dataset generation script
├── explainable_ai.py               # Explainable AI module
├── requirements.txt                # Python dependencies
│
├── dataset/
│   └── carbon_footprint_dataset.csv  # Training dataset (10,000 samples, 31 features)
│
├── models/
│   ├── carbon_regressor.pkl         # Best regression model (XGBoost)
│   ├── carbon_classifier.pkl       # Emission category classifier
│   ├── encoder.pkl                 # Fuel type label encoder
│   ├── scaler.pkl                  # Feature scaler
│   ├── model_metadata.pkl          # Regression model metadata
│   └── classifier_metadata.pkl     # Classification model metadata
│
├── preprocessing/
│   └── preprocess.py               # Data preprocessing pipeline
│
├── recommendation_engine/
│   └── recommendations.py          # Personalized recommendation engine
│
├── templates/
│   └── index.html                  # Modern web interface with Tabler Icons
│
└── static/
    └── hero.png                    # Hero image for web interface
```

## 🚀 Features

### Machine Learning Models
- **Regression**: XGBoost Regressor (R² Score: 0.9492)
- **Classification**: Random Forest Classifier (Accuracy: 80.6%)
- **Model Comparison**: Linear Regression, Random Forest, XGBoost

### Explainable AI
- **Feature Importance**: Global feature contribution analysis
- **SHAP Explanations**: Local prediction explanations
- **Top Contributors**: Identifies key emission sources

### Recommendation Engine
- **Category-Based**: Recommendations based on emission level
- **Feature-Specific**: Targeted suggestions for high-impact areas
- **Priority Actions**: Urgent vs. long-term recommendations

### Web Interface
- **Modern Design**: Clean UI with Tabler Icons and hero section
- **User-Friendly Form**: 31 input fields organized into 5 categories
- **Real-Time Predictions**: Instant ML-based results
- **Visual Insights**: Color-coded categories, progress bars, and sustainability score
- **Personalized Advice**: Actionable recommendations with priority actions

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 500MB disk space

## 🔧 Installation

### Step 1: Clone or Download Project

```bash
cd "c:\Users\pc\OneDrive\Desktop\carbon Footprint"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train ML Models

#### Option A: Quick Start (Use Pre-trained Models)

If models are already in the `models/` directory, skip to Step 4.

#### Option B: Train from Scratch

```bash
# Generate training dataset
python generate_dataset.py

# Train regression model
python train_model.py

# Train classification model
python train_classifier.py
```

**Expected Output:**
- Dataset: 10,000 samples with 31 features
- Best Model: XGBoost (R²: 0.9492)
- Classifier Accuracy: 80.6%

### Step 4: Run Flask Application

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000/`

## 🎯 Usage

### Web Interface

1. Open browser and navigate to `http://127.0.0.1:5000/`
2. Enter your lifestyle data across 5 categories:
   - **Transportation**: Car travel, fuel type, public transport, train travel, bike usage, cab/rideshare, vehicle type, vehicle age, number of vehicles
   - **Household Energy**: Electricity consumption, household members, AC usage, LPG consumption, renewable energy, solar panels, home type
   - **Food & Diet**: Diet type, meat meals, dairy consumption, packaged food, food waste, local food consumption
   - **Travel**: Flight hours, domestic flights, international flights, hotel stays, vacation frequency
   - **Consumer Behavior**: Online shopping, fast fashion, electronics, recycling score, waste segregation
3. Click "Calculate carbon footprint"
4. View AI-powered results:
   - Predicted annual carbon footprint
   - Emission category (Low/Medium/High/Critical)
   - Sustainability score (out of 100)
   - Category breakdown with percentages
   - Top emission contributors with progress bars
   - Priority actions
   - Personalized recommendations



## 📊 Model Performance

### Regression Models

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 435.12 | 652.10 | 0.9446 |
| Random Forest | 534.14 | 768.26 | 0.9231 |
| **XGBoost** | **447.97** | **624.08** | **0.9492** |

### Classification Model

- **Model**: Random Forest Classifier
- **Accuracy**: 80.6%
- **Categories**: Low, Medium, High, Critical

### Emission Category Thresholds

- **Low**: ≤ 7,128 kg CO₂/year
- **Medium**: 7,128 - 8,899 kg CO₂/year
- **High**: 8,899 - 10,831 kg CO₂/year
- **Critical**: > 10,831 kg CO₂/year

## 🔬 Technical Details

### Feature Engineering

- **Base Features**: 31 features across 5 categories (Transportation, Household Energy, Food & Diet, Travel, Consumer Behavior)
- **Engineered Features**: 8 interaction and ratio features (e.g., electricity_per_member, transport_emission_score, diet_emission_score, travel_emission_score, consumption_emission_score, total_transport_score, energy_efficiency_score, sustainable_living_score)
- **Total Features**: 40 features (31 base + 8 engineered)

### Preprocessing Pipeline

1. Missing value handling (median/mode imputation)
2. Label encoding for categorical variables
3. Feature scaling (StandardScaler)
4. Feature engineering
5. Train/test split (80/20)

### Model Selection

- **Regression**: XGBoost selected based on highest R² score
- **Classification**: Random Forest for interpretability and performance

## 🛠️ Model Training Instructions

### Retrain Regression Model

```bash
python train_model.py
```

This will:
1. Load dataset from `dataset/carbon_footprint_dataset.csv`
2. Preprocess data
3. Train Linear Regression, Random Forest, and XGBoost
4. Evaluate using MAE, RMSE, and R²
5. Select best model automatically
6. Save to `models/carbon_regressor.pkl`

### Retrain Classification Model

```bash
python train_classifier.py
```

This will:
1. Load dataset and create emission categories
2. Preprocess data
3. Train Random Forest Classifier
4. Evaluate using accuracy and classification report
5. Save to `models/carbon_classifier.pkl`

### Generate New Dataset

```bash
python generate_dataset.py
```

This will:
1. Generate 10,000 realistic samples with 31 features
2. Use carbon emission formulas with realistic distributions
3. Save to `dataset/carbon_footprint_dataset.csv`

## 📝 API Reference

### Flask Endpoints

- **GET /**: Render calculator form
- **POST /**: Process form and return predictions

### Key Classes

#### `CarbonFootprintPreprocessor`
- `preprocess_for_training(df)`: Prepare data for model training
- `preprocess_for_prediction(input_data)`: Prepare single input for prediction
- `save_preprocessors(output_dir)`: Save encoder and scaler
- `load_preprocessors(input_dir)`: Load encoder and scaler

#### `ExplainableAI`
- `get_feature_importance()`: Get global feature importance
- `get_shap_explanation(input_data)`: Get SHAP values for prediction
- `get_top_contributors(input_data, top_n=4)`: Get top N contributors
- `get_category_breakdown(input_data)`: Get category-wise emission breakdown

#### `RecommendationEngine`
- `get_category_recommendations(category)`: Get category-based recommendations
- `get_feature_specific_recommendations(top_contributors)`: Get feature-specific recommendations
- `get_personalized_recommendations(category, top_contributors, category_breakdown)`: Get complete recommendations
- `get_priority_actions(category, top_contributors)`: Get priority actions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available for educational and research purposes.

## Acknowledgments

- XGBoost library for gradient boosting
- SHAP library for explainable AI
- Scikit-learn for ML algorithms
- Flask for web framework


## 🌱 Environmental Impact

This project aims to help individuals understand and reduce their carbon footprint through data-driven insights and personalized recommendations. By making informed choices, users can contribute to environmental sustainability.

---

**Version**: 1.0.0  
**Last Updated**: June 2026  
**Status**: Production Ready
