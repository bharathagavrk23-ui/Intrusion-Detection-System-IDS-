import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("sample_nsl_kdd.csv")

# Encode categorical columns
categoricals = ['protocol_type', 'service', 'flag']
for col in categoricals:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Simplify label to 'normal' or 'attack'
df['label'] = df['label'].apply(lambda x: 'attack' if x != 'normal' else 'normal')

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "ids_model.pkl")
print("✅ Model saved as ids_model.pkl")
