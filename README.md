# 🔐 Intrusion Detection System (IDS) Web App using Flask and Machine Learning

This project is a web-based Intrusion Detection System (IDS) built using Flask and a pre-trained machine learning model. It allows users to upload NSL-KDD formatted network traffic data and receive real-time predictions identifying potential attacks.

## 🛠 Features

- 📁 Upload `.csv` files containing network traffic data.
- 🤖 Applies preprocessing (categorical encoding) to prepare data for model inference.
- 🧠 Utilizes a trained ML model (`ids_model.pkl`) to classify traffic as *normal* or *attack*.
- 📊 Displays the top predictions and provides a downloadable CSV of the results.
- 🖥️ Built with Flask for web deployment and Python's data science stack (pandas, scikit-learn, joblib).

## 📂 Files Included

- `web_app.py`: Flask web app that handles file uploads, preprocessing, and displays results.
- `predict.py`: Standalone script for local batch predictions on CSV data.
- `sample_nsl_kdd.csv`: Example input file for testing.
- `ids_model.pkl`: Pre-trained intrusion detection model (not included here due to size).

## 🚀 How to Run

1. **Install dependencies**:
   pip install flask pandas scikit-learn joblib
   
2. Place ids_model.pkl in the project root directory.
3. Run the web app:
    python web_app.py



