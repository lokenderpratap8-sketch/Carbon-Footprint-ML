# Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)
```bash
python generate_dataset.py
python train_model.py
python train_classifier.py
```

### 3. Run Application
```bash
python app.py
```

Open browser: http://127.0.0.1:5000/

## Detailed Setup Instructions

### Prerequisites Check

```bash
python --version  # Should be 3.8 or higher
pip --version     # Should be installed
```

### Step-by-Step Installation

#### Step 1: Verify Project Structure

Ensure you have the following directories:
```
carbon Footprint/
├── dataset/
├── models/
├── preprocessing/
├── recommendation_engine/
├── templates/
└── static/
```

#### Step 2: Install Python Packages

```bash
pip install flask pandas numpy scikit-learn xgboost shap joblib
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

#### Step 3: Generate Training Dataset

```bash
python generate_dataset.py
```

Expected output:
```
Dataset generated successfully!
Total samples: 5000
Dataset saved to: dataset/carbon_footprint_dataset.csv
```

#### Step 4: Train Regression Model

```bash
python train_model.py
```

Expected output:
```
BEST MODEL: XGBoost
R² Score: 0.9492
✓ Best model saved to models/carbon_regressor.pkl
```

#### Step 5: Train Classification Model

```bash
python train_classifier.py
```

Expected output:
```
Accuracy: 0.8060
✓ Classifier saved to models/carbon_classifier.pkl
```

#### Step 6: Verify Model Files

Check that these files exist in `models/` directory:
- carbon_regressor.pkl
- carbon_classifier.pkl
- encoder.pkl
- scaler.pkl
- model_metadata.pkl
- classifier_metadata.pkl

#### Step 7: Run Flask Application

```bash
python app.py
```

Expected output:
```
Loading ML models...
✓ All ML models loaded successfully
 * Running on http://127.0.0.1:5000
```

#### Step 8: Test Application

1. Open browser to http://127.0.0.1:5000/
2. Enter sample data:
   - Electricity: 400
   - Car KM: 500
   - Fuel Type: Petrol
   - Flight Hours: 10
   - Meat Meals: 7
3. Click "Calculate"
4. Verify ML results appear:
   - AI-Powered Insights section
   - Top Emission Contributors
   - Priority Actions
   - Personalized Recommendations

## Troubleshooting

### Issue: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**:
```bash
pip install xgboost
```

### Issue: Models not loading

**Error**: `ML models not loaded. Please run training scripts first.`

**Solution**:
```bash
python generate_dataset.py
python train_model.py
python train_classifier.py
```

### Issue: FileNotFoundError

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/carbon_footprint_dataset.csv'`

**Solution**:
```bash
python generate_dataset.py
```

### Issue: Port already in use

**Error**: `Address already in use`

**Solution**:
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

### Issue: SHAP errors

**Error**: SHAP-related errors during prediction

**Solution**: The system will fall back to feature importance if SHAP fails. This is expected behavior.

## Development Setup

### Running Individual Components

#### Test Preprocessing
```bash
python preprocessing/preprocess.py
```

#### Test Explainable AI
```bash
python explainable_ai.py
```

#### Test Recommendation Engine
```bash
python recommendation_engine/recommendations.py
```

### Retraining Models

To retrain with new data:

1. Update or regenerate dataset:
```bash
python generate_dataset.py
```

2. Train regression model:
```bash
python train_model.py
```

3. Train classification model:
```bash
python train_classifier.py
```

4. Restart Flask app:
```bash
python app.py
```

## Production Deployment

### For Local Deployment

The current setup is ready for local use. To deploy:

1. Set debug mode to False in app.py:
```python
app.run(debug=False)
```

2. Use a production WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### For Cloud Deployment

#### Heroku
1. Create Procfile:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create
git push heroku main
```

#### Docker
Create Dockerfile:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t carbon-footprint .
docker run -p 5000:5000 carbon-footprint
```

## Performance Optimization

### Model Loading Optimization

Models are loaded once at startup. For large-scale deployments:

1. Use model compression
2. Implement model versioning
3. Add caching layer
4. Use async processing

### Database Integration (Optional)

To store predictions and user data:

1. Add database connection in app.py
2. Store predictions with timestamps
3. Enable historical tracking
4. Add user authentication

## Security Considerations

1. **Input Validation**: All inputs are validated in Flask
2. **Model Security**: Models are loaded from local files
3. **No External APIs**: All processing is local
4. **No User Data Storage**: No personal data is persisted

## Maintenance

### Regular Tasks

- Retrain models quarterly with new data
- Update dependencies annually
- Monitor model performance
- Review and update recommendations

### Model Monitoring

Track:
- Prediction accuracy
- User engagement
- Recommendation effectiveness
- System performance

## Support

For issues:
1. Check this SETUP.md
2. Review README.md
3. Verify all steps completed
4. Check error messages in terminal

---

**Last Updated**: June 2026
