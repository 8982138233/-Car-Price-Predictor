import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

# Load the cleaned car dataset
car = pd.read_csv("Cleaned_Car.csv")

@app.route('/')
def index():
    """Render the index page with dropdown options."""
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Add a default option for company
    companies.insert(0, "Select Company")

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request and return the result."""
    # Retrieve form data
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # Make a prediction using the loaded model
    prediction = model.predict(pd.DataFrame([
        [car_model, company, year, kms_driven, fuel_type]
    ], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
