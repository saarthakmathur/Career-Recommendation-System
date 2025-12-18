import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("All libraries imported successfully!")



np.random.seed(42)
n_samples = 250

careers = [
    "Software Engineer",
    "Data Analyst",
    "Web Developer",
    "ML Engineer",
    "UI/UX Designer",
    "Project Manager"
]

dummy_data = {
    'logical_reasoning': np.random.randint(3, 10, n_samples),
    'programming_knowledge': np.random.randint(1, 10, n_samples),
    'maths_score': np.random.randint(2, 10, n_samples),
    'communication_skills': np.random.randint(1, 10, n_samples),
    'creativity': np.random.randint(1, 10, n_samples),
    'management_skills': np.random.randint(1, 10, n_samples),
    'academic_percentage': np.random.randint(50, 95, n_samples),
    'interest_technology': np.random.randint(1, 10, n_samples),
    'interest_management': np.random.randint(1, 10, n_samples),
    'interest_design': np.random.randint(1, 10, n_samples),
}

df = pd.DataFrame(dummy_data)



def assign_career(row):
    if row['programming_knowledge'] > 7 and row['interest_technology'] > 7:
        return "Software Engineer"
    elif row['maths_score'] > 7 and row['logical_reasoning'] > 7:
        return "Data Analyst"
    elif row['creativity'] > 7 and row['interest_design'] > 7:
        return "UI/UX Designer"
    elif row['management_skills'] > 7 and row['communication_skills'] > 7:
        return "Project Manager"
    else:
        return np.random.choice(careers)

df['career'] = df.apply(assign_career, axis=1)

df.to_csv("career_data.csv", index=False)
print("Dummy dataset created successfully!\n")
print(df.head())



data = pd.read_csv("career_data.csv")
print("\nDataset Loaded Successfully!")
display(data.head())
print("\nShape:", data.shape)
print("\nCareer Distribution:\n", data['career'].value_counts())



target_col = 'career'
feature_cols = [col for col in data.columns if col != target_col]

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)



dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

print("\nAll models trained successfully!")



models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model
}

accuracy_results = {}

print("\n===== MODEL EVALUATIONS =====")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc

    print(f"\n{name} Accuracy: {round(acc * 100, 2)}%")
    print(classification_report(y_test, y_pred))



plt.figure(figsize=(6,4))
plt.bar(accuracy_results.keys(), accuracy_results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
for i, v in enumerate(accuracy_results.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()



best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

print("\nBest Model =", best_model_name)

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_,
            cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



importances = rf_model.feature_importances_
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_df)
plt.title("Feature Importance (Random Forest)")
plt.show()



print("\n===== SAMPLE CAREER PREDICTION =====")

new_student = {
    'logical_reasoning': 8,
    'programming_knowledge': 9,
    'maths_score': 9,
    'communication_skills': 6,
    'creativity': 5,
    'management_skills': 4,
    'academic_percentage': 82,
    'interest_technology': 9,
    'interest_management': 4,
    'interest_design': 3
}

new_df = pd.DataFrame([new_student])
pred = best_model.predict(new_df)[0]
confidence = best_model.predict_proba(new_df).max()

print("\nInput Student Data:")
display(new_df)

print("\n🎯 Recommended Career:", pred)
print("Confidence Score:", round(confidence * 100, 2), "%")

