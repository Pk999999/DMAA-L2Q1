import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('diet maintenance.csv')

X = df.drop('StrictDietNeeded', axis=1)
y = df['StrictDietNeeded']
le_gender = LabelEncoder()
le_activity = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])
X['ActivityLevel'] = le_activity.fit_transform(X['ActivityLevel'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
def predict_strict_diet(patient_data):
    patient_df = pd.DataFrame([patient_data])
    patient_df['Gender'] = le_gender.transform(patient_df['Gender'])
    patient_df['ActivityLevel'] = le_activity.transform(patient_df['ActivityLevel'])
    prediction = nb_classifier.predict(patient_df)
    return prediction[0]

new_patient = {
    'Gender': 'Female',
    'Age': 35,
    'Weight': 70.5,
    'Height': 165.0,
    'BMI': 25.9,
    'ActivityLevel': 'Moderate',
    'PreviousDietAttempts': 2
}

result = predict_strict_diet(new_patient)
print(f"\n Prithvi Kathuria(21BBS0158),Strict diet recommendation for the new patient: {result}")