from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
# Pre-defined encoding for email model only

# Load all models
def load_models():
    models = {}
    try:
        with open('model/animal_model.pkl', 'rb') as f:
            models['animal'] = pickle.load(f)
        with open('model/email_model.pkl', 'rb') as f:
            models['email'] = pickle.load(f)
        with open('model/loan_model.pkl', 'rb') as f:
            models['loan'] = pickle.load(f)
        with open('model/weather_model.pkl', 'rb') as f:
            models['weather'] = pickle.load(f)
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    return models

models = load_models()
le = LabelEncoder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<model_type>', methods=['GET', 'POST'])
def predict(model_type):
    if request.method == 'POST':
        try:
            if model_type == 'animal':
                # Animal prediction logic
                animal = request.form['animal']
                size = request.form['size']
                color = request.form['color']
                
                # Encode inputs
                animal_encoded = le.fit_transform([animal])[0]
                size_encoded = le.fit_transform([size])[0]
                color_encoded = le.fit_transform([color])[0]
                
                prediction = models['animal'].predict([[animal_encoded, size_encoded, color_encoded]])
                result = "Yes" if prediction[0] == 1 else "No"
                return render_template('prediction.html', 
                                     model_type=model_type,
                                     result=result,
                                     input_data=f"Animal: {animal}, Size: {size}, Color: {color}")
            
            elif model_type == 'email':
                # Email prediction logic
                offer = request.form['offer']
                link = request.form['link']
                greeting = request.form['greeting']
                sender = request.form['sender']
                
                # Encode inputs
                offer_encoded = le.fit_transform([offer])[0]
                link_encoded = le.fit_transform([link])[0]
                greeting_encoded = le.fit_transform([greeting])[0]
                sender_encoded = le.fit_transform([sender])[0]
                
                prediction = models['email'].predict([[offer_encoded, link_encoded, greeting_encoded, sender_encoded]])
                result = "Spam" if prediction[0] == 1 else "Not Spam"
                return render_template('prediction.html',
                                     model_type=model_type,
                                     result=result,
                                     input_data=f"Offer: {offer}, Link: {link}, Greeting: {greeting}, Sender Known: {sender}")
            
            elif model_type == 'loan':
                # Loan prediction logic
                age = request.form['age']
                income = request.form['income']
                credit = request.form['credit']
                employment = request.form['employment']
                
                # Encode inputs
                age_encoded = le.fit_transform([age])[0]
                income_encoded = le.fit_transform([income])[0]
                credit_encoded = le.fit_transform([credit])[0]
                employment_encoded = le.fit_transform([employment])[0]
                
                prediction = models['loan'].predict([[age_encoded, income_encoded, credit_encoded, employment_encoded]])
                result = "Approved" if prediction[0] == 1 else "Rejected"
                return render_template('prediction.html',
                                     model_type=model_type,
                                     result=result,
                                     input_data=f"Age: {age}, Income: {income}, Credit: {credit}, Employment: {employment}")
            
            elif model_type == 'weather':
                # Weather prediction logic
                weather = request.form['weather']
                road = request.form['road']
                traffic = request.form['traffic']
                engine = request.form['engine']
                
                # Encode inputs
                weather_encoded = le.fit_transform([weather])[0]
                road_encoded = le.fit_transform([road])[0]
                traffic_encoded = le.fit_transform([traffic])[0]
                engine_encoded = le.fit_transform([engine])[0]
                
                prediction = models['weather'].predict([[weather_encoded, road_encoded, traffic_encoded, engine_encoded]])
                result = "Accident Likely" if prediction[0] == 1 else "Accident Unlikely"
                return render_template('prediction.html',
                                     model_type=model_type,
                                     result=result,
                                     input_data=f"Weather: {weather}, Road: {road}, Traffic: {traffic}, Engine Problem: {engine}")
        
        except Exception as e:
            return render_template('prediction.html', 
                                 error=f"Prediction error: {str(e)}")
    
    # GET request - show form
    return render_template('prediction.html', model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)