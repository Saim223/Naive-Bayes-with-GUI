# Naive-Bayes-with-GUI
A Flask-based machine learning web app using multiple trained models for predictions, including animals, weather, spam detection, and loans.

ML Prediction Web App

This project is a Flask-based machine learning web application that performs multiple predictions using pre-trained models, including:

Animal prediction model

Weather condition prediction model

Email spam detection model

Loan approval prediction model

The app includes a frontend interface (HTML templates + CSS/JS), a backend API in Flask, and several .pkl machine-learning models stored in the model/ directory.

📌 Project Structure
New folder/
│── app.py
│── requirements.txt
│── train_animal_model.py
│
├── model/
│   ├── animal_model.pkl
│   ├── email_model.pkl
│   ├── loan_model.pkl
│   ├── weather_model.pkl
│   ├── AnimalInformation.csv
│   ├── EmailSpamDetectionUpdated.csv
│   └── weatherAndRoadCondition.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── templates/
    ├── base.html
    ├── index.html
    └── prediction.html

🚀 Features

Clean UI created with HTML, CSS, and JS

Flask backend with complete routing

Multiple ML models for predictions

CSV datasets included

Ready-to-deploy project

📦 Installation
1. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows

2. Install dependencies
pip install -r requirements.txt

▶️ Running the App
python app.py


Then open in your browser:

http://127.0.0.1:5000/

📁 Models

All trained machine learning models are stored in the model/ folder:

animal_model.pkl

email_model.pkl

weather_model.pkl

loan_model.pkl

⚠️ IMPORTANT — FOR ANYONE USING THIS CODE

If you clone, copy, or reuse this project:

👉 You MUST replace your ML models with the models provided in my model/ folder.

The app is trained and designed to work only with these model files, so make sure you use:

model/animal_model.pkl
model/email_model.pkl
model/weather_model.pkl
model/loan_model.pkl


Otherwise, the predictions will not work correctly.

📚 Training (Optional)

The repository includes training scripts such as:

train_animal_model.py

You may retrain the models or create new ones, but if you do:

➡️ Update the model files in the model/ directory with your new ones.

📝 License

This project is open for use, modification, and learning purposes.
If you use this code, please credit the original author.

🙌 Author

Hafiz Saim Murtaza
