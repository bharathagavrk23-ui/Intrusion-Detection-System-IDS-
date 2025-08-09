
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load("ids_model.pkl")

# Load new data (should have the same structure as training data)
df = pd.read_csv("sample_nsl_kdd.csv")

# Encode categorical features
categoricals = ['protocol_type', 'service', 'flag']
for col in categoricals:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Simplify label (if present)
if 'label' in df.columns:
    df['label'] = df['label'].apply(lambda x: 'attack' if x != 'normal' else 'normal')
    y = df['label']
    X = df.drop('label', axis=1)
else:
    X = df

# Predict
predictions = model.predict(X)
df['Prediction'] = predictions

# Output results
df.to_csv("predicted_output.csv", index=False)
print("✅ Predictions saved to predicted_output.csv")
