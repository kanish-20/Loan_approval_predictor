# Bank Loan Approval Predictor

A Flask-based machine learning web app that predicts whether a bank loan should be approved or not based on user input.

---

# Features
Predicts loan approval using Logistic Regression

Trained on user-generated dataset

Web interface to collect applicant details

Styled with basic HTML + CSS


# How It Works

The model is trained on bank loan applicant data.

### User enters:

Gender

Marital Status

Applicant Income

Loan Amount

Credit History

### The model uses Logistic Regression to predict:

✅ Loan Approved

❌ Loan Not Approved

Result is displayed on the screen.


# Tech Stack

Python

Flask

Pandas

scikit-learn

HTML/CSS


#Project Structure

```
Loan_approval_predictor/
│
├── app.py                 # Flask app
├── train_model.py         # Model training
├── requirements.txt       # Dependencies
├── scaler.pkl             # Saved StandardScaler
├── loan_model.pkl         # Trained ML model
├── templates/
│   └── index.html         # HTML Form UI

```

# How to Run Locally
### 1.Clone the repo
``git clone https://github.com/kanish-20/Loan_approval_predictor.git
cd Loan_approval_predictor
``

### 2.Install dependencies
```
pip install -r requirements.txt
```

### 3.Run the app
```
python app.py
```

### 4.Open browser
```
http://127.0.0.1:5000/
```

----

# Future Improvements

-> Add more features like education, dependents, employment

-> Store approved/rejected results in a database

-> Deploy to Render or Railway for public access

----

