import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("diabetes.csv")

print("\nFirst 5 Records")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nStatistical Summary")
print(data.describe())

print("\nMissing Values")
print(data.isnull().sum())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n================================")
print("LOGISTIC REGRESSION RESULTS")
print("================================")

print("Accuracy:", lr_accuracy)

lr_cm = confusion_matrix(y_test, lr_pred)

print("\nConfusion Matrix")
print(lr_cm)

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

plt.figure(figsize=(6, 4))

sns.heatmap(
    lr_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-Diabetic', 'Diabetic'],
    yticklabels=['Non-Diabetic', 'Diabetic']
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("logistic_regression_cm.png")
plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n================================")
print("RANDOM FOREST RESULTS")
print("================================")

print("Accuracy:", rf_accuracy)

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix")
print(rf_cm)

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

plt.figure(figsize=(6, 4))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Non-Diabetic', 'Diabetic'],
    yticklabels=['Non-Diabetic', 'Diabetic']
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("random_forest_cm.png")
plt.show()

models = ["Logistic Regression", "Random Forest"]
accuracies = [lr_accuracy, rf_accuracy]

plt.figure(figsize=(6, 4))

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.2f}",
        ha='center'
    )

plt.savefig("accuracy_comparison.png")
plt.show()

if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("\nBest Model: Random Forest")
else:
    best_model = lr_model
    print("\nBest Model: Logistic Regression")

joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully")

new_patient = pd.DataFrame(
    [[2, 120, 70, 25, 100, 28.5, 0.5, 35]],
    columns=X.columns
)

new_patient_scaled = scaler.transform(new_patient)

prediction = best_model.predict(new_patient_scaled)

print("\nNew Patient Prediction")

if prediction[0] == 1:
    print("Patient is Diabetic")
else:
    print("Patient is Non-Diabetic")