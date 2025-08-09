
from flask import Flask, request, render_template, redirect
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
model = joblib.load("ids_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)

    # Encode categorical columns
    for col in ['protocol_type', 'service', 'flag']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df.drop('label', axis=1) if 'label' in df.columns else df
    preds = model.predict(X)

    df['Prediction'] = preds
    output_path = "static/predicted_output.csv"
    df.to_csv(output_path, index=False)

    return render_template('result.html', table=df.head().to_html(classes='table table-striped', index=False), download_link=output_path)

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
