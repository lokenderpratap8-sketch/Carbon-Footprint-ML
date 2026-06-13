import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Set random seed for reproducibility
np.random.seed(42)

# Generate realistic synthetic data
n_samples = 5000

# Monthly electricity consumption (kWh) - typical range: 100-1000 kWh/month
electricity = np.random.normal(loc=400, scale=150, size=n_samples)
electricity = np.clip(electricity, 50, 1200)

# Monthly car travel distance (km) - typical range: 0-3000 km/month
car_km = np.random.exponential(scale=500, size=n_samples)
car_km = np.clip(car_km, 0, 5000)

# Fuel type distribution
fuel_types = ['petrol', 'diesel', 'none']
fuel_type = np.random.choice(fuel_types, size=n_samples, p=[0.5, 0.4, 0.1])

# Flight hours per year - typical range: 0-100 hours
flight_hours = np.random.exponential(scale=10, size=n_samples)
flight_hours = np.clip(flight_hours, 0, 150)

# Meat meals per week - typical range: 0-21 meals
meat_meals = np.random.normal(loc=7, scale=5, size=n_samples)
meat_meals = np.clip(meat_meals, 0, 21)

# Calculate annual carbon emission using realistic formulas
# Electricity: 0.92 kg CO2 per kWh (monthly * 12)
annual_electricity = electricity * 12 * 0.92

# Car: Petrol 0.192 kg CO2/km, Diesel 0.171 kg CO2/km (monthly * 12)
annual_car = np.zeros(n_samples)
for i in range(n_samples):
    if fuel_type[i] == 'petrol':
        annual_car[i] = car_km[i] * 12 * 0.192
    elif fuel_type[i] == 'diesel':
        annual_car[i] = car_km[i] * 12 * 0.171
    else:
        annual_car[i] = 0

# Flights: 90 kg CO2 per hour
annual_flights = flight_hours * 90

# Meat: 7.2 kg CO2 per meal (weekly * 52)
annual_meat = meat_meals * 52 * 7.2

# Total annual carbon emission (kg CO2)
annual_carbon_emission = annual_electricity + annual_car + annual_flights + annual_meat

# Add some realistic noise to make it more ML-friendly
noise = np.random.normal(loc=0, scale=annual_carbon_emission * 0.05, size=n_samples)
annual_carbon_emission = annual_carbon_emission + noise
annual_carbon_emission = np.clip(annual_carbon_emission, 0, None)

# Create DataFrame
df = pd.DataFrame({
    'electricity': electricity,
    'car_km': car_km,
    'fuel_type': fuel_type,
    'flight_hours': flight_hours,
    'meat_meals': meat_meals,
    'annual_carbon_emission': annual_carbon_emission
})

# Shuffle the dataset
df = shuffle(df, random_state=42)

# Save to CSV
df.to_csv('dataset/carbon_footprint_dataset.csv', index=False)

print(f"Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"\nDataset statistics:")
print(df.describe())
print(f"\nFuel type distribution:")
print(df['fuel_type'].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset saved to: dataset/carbon_footprint_dataset.csv")
