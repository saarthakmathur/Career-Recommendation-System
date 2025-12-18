# 🎯 Career Recommendation System using Machine Learning

A Machine Learning–based system that predicts the most suitable career for a student based on skills, academic performance, and interests using classification algorithms.

---

## 📌 Project Description

Choosing the right career is a critical decision for students.  
This project uses Machine Learning models to analyze student attributes and recommend an appropriate career path. The system compares multiple classification models and selects the best-performing one based on accuracy.

This project is developed for **academic learning and portfolio demonstration purposes**.

---

## 🧠 Machine Learning Models Used

- Decision Tree Classifier  
- Random Forest Classifier  
- Logistic Regression (Multiclass)

The model with the highest accuracy is selected automatically.

---

## 📊 Dataset Information

- Dataset Type: Synthetic (Dummy data generated using NumPy)
- Total Samples: 250
- Target Variable: `career`

### Career Classes
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

## 🔁 Project Workflow

1. Generate synthetic student data  
2. Assign career labels using rule-based logic  
3. Split data into training and testing sets  
4. Train multiple ML models  
5. Evaluate models using accuracy and classification report  
6. Compare model performance  
7. Select the best model  
8. Predict career for a new student  

---

## 📈 Model Evaluation

- Accuracy Score  
- Classification Report  
- Confusion Matrix  
- Feature Importance (Random Forest)  
- Visual comparison of model accuracies  

---

## 🔮 Sample Output

🎯 Recommended Career: Software Engineer
Confidence Score: 92.34%


---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/saarthakmathur/Career-Recommendation-System.git
cd Career-Recommendation-System

## 2️⃣ Install Dependencies
pip install -r requirements.txt

## 3️⃣ Run the Project
python career_prediction.py

## 📌 Future Improvements

Use real-world student datasets

Add Deep Learning models

Build a web interface using Flask or Streamlit

Improve prediction confidence calibration

Add more career categories

## ⚠️ Disclaimer

This project uses synthetic data and is intended for educational and portfolio purposes only.
Predictions should not be considered real career advice.

## 👤 Author

Saarthak Mathur
