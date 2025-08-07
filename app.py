from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        gender = 1 if request.form['Gender'] == 'Male' else 0
        married = 1 if request.form['Married'] == 'Yes' else 0
        income = float(request.form['ApplicantIncome'])
        loan = float(request.form['LoanAmount'])
        credit = int(request.form['Credit_History'])
        features = np.array([[gender, married, income, loan, credit]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        result = "Loan Approved" if prediction == 1 else "Loan Rejected"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
