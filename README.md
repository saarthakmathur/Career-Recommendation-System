# 🎯 Career Recommendation System using Machine Learning

This project predicts the most suitable career for a student based on academic performance, skills, and interests using Machine Learning classification models.

---

## 📌 Project Overview

Choosing the right career is a challenging task for students.  
This system uses **Machine Learning algorithms** to recommend a career path based on multiple attributes such as:

- Programming knowledge  
- Logical reasoning  
- Mathematical ability  
- Communication skills  
- Creativity  
- Interests (Technology, Design, Management)

---

## 🧠 Machine Learning Models Used

- Decision Tree Classifier  
- Random Forest Classifier  
- Logistic Regression (Multiclass)

The best-performing model is automatically selected based on accuracy.

---

## 📊 Dataset

- Synthetic (dummy) dataset generated using NumPy  
- Total samples: **250**
- Target variable: `career`
- Career categories:
  - Software Engineer
  - Data Analyst
  - Web Developer
  - ML Engineer
  - UI/UX Designer
  - Project Manager

---

## ⚙️ Features Used

- logical_reasoning  
- programming_knowledge  
- maths_score  
- communication_skills  
- creativity  
- management_skills  
- academic_percentage  
- interest_technology  
- interest_management  
- interest_design  

---

## 🔁 Workflow

1. Generate synthetic dataset  
2. Assign career labels using rule-based logic  
3. Train ML models  
4. Evaluate using:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
5. Compare model performance  
6. Predict career for a new student  

---

## 📈 Model Evaluation

- Accuracy comparison between models
- Confusion Matrix visualization
- Feature importance using Random Forest

---

## 🔮 Sample Prediction

```python
🎯 Recommended Career: Software Engineer  
Confidence Score: 92.34%

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🚀 How to Run the Project
pip install -r requirements.txt
python career_prediction.py

Future Improvements

Use real student data

Add Deep Learning model

Build a web interface using Flask / Streamlit

Improve confidence calibration
