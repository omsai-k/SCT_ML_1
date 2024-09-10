from flask import Flask, request, render_template
import pickle as pkl
import numpy as np

app = Flask(__name__)

with open('HousePricePredictor','rb') as file:
    model = pkl.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    square_footage = float(request.form['square_footage'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    
    features = np.array([[square_footage,bedrooms,bathrooms]])
    
    predicted_price = model.predict(features)[0]
    
    return render_template('index.html',prediction_text=f"Predicted House Price : {predicted_price:.2f} USD",
                           square_footage=square_footage,
                           bedrooms=bedrooms,
                           bathrooms=bathrooms)

if __name__ == "__main__":
    app.run(debug=True)