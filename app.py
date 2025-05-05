from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('diabetes_rf_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

# Feature names in the correct order
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make prediction"""
    try:
        # Get form data and convert to float
        features = [float(request.form[feature]) for feature in feature_names]
        
        # Create DataFrame with correct feature order
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]  # Probability of diabetes
        
        # Prepare result
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'interpretation': 'Likely Diabetic' if prediction[0] == 1 else 'Likely Non-Diabetic'
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        data = request.get_json()
        
        # Validate all required features are present
        if not all(feature in data for feature in feature_names):
            return jsonify({'error': 'Missing required features'}), 400
            
        # Prepare input data
        features = [float(data[feature]) for feature in feature_names]
        input_data = pd.DataFrame([features], columns=feature_names)
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'interpretation': 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)