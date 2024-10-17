import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
 
# Load dataset
data_path = "/Users/lyricabdul-rasheed/Downloads/wildfire_prediction_multi_output_dataset_v2.xlsx"

df = pd.read_excel(data_path)

# Print column names for debugging
print("Column Names: ", df.columns)

# Clean column names
df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace

# Update required columns to match actual column names
required_columns = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 
                    'Rainfall (mm)', 'Fuel Moisture (%)', 'Vegetation Type', 
                    'Slope (%)', 'Region', 'Fire Size (hectares)', 
                    'Fire Duration (hours)', 'Suppression Cost ($)', 'Fire Occurrence']

# Check if required columns are present
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Check for and handle missing values
if df.isnull().values.any():
    print("Missing values detected. Filling with median values for numerical columns and mode for categorical columns.")
    for col in df.columns:
        if df[col].dtype == np.number:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Preprocessing
df['Vegetation Type'] = df['Vegetation Type'].astype('category')
df['Region'] = df['Region'].astype('category')

label_encoders = {}
for column in ['Vegetation Type', 'Region']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target variables
X = df[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 
         'Rainfall (mm)', 'Fuel Moisture (%)', 'Vegetation Type', 
         'Slope (%)', 'Region']]
y = df[['Fire Size (hectares)', 'Fire Duration (hours)', 'Suppression Cost ($)']]

# Check for NaN values in the target variables
if y.isnull().values.any():
    raise ValueError("Target variable contains NaN values. Please clean your data.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'wildfire_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model trained and saved successfully!")

# Function to make predictions
def predict_wildfire(input_data):
    # Load the model and label encoders
    model = joblib.load('wildfire_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')

    # Encode categorical inputs
    for column in ['Vegetation Type', 'Region']:
        input_data[column] = label_encoders[column].transform([input_data[column]])[0]

    # Make prediction
    prediction = model.predict(np.array(list(input_data.values())).reshape(1, -1))
    return prediction

# Example input for prediction
input_data = {
    'Temperature (°C)': 30,
    'Humidity (%)': 40,
    'Wind Speed (km/h)': 15,
    'Rainfall (mm)': 0,
    'Fuel Moisture (%)': 12,
    'Vegetation Type': 'Forest',
    'Slope (%)': 10,
    'Region': 'North'
}

predictions = predict_wildfire(input_data)
print("Predictions (Fire Size, Fire Duration, Suppression Cost): ", predictions)
