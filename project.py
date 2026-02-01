from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('svm (2).pkl')

@app.route('/')
def home():
    return render_template('proj.html')
@app.route('/predict', methods=['GET'])
def predict():
    # Get input values from form
    ApplicantIncome = int(request.form['applicantincome'])
    LoanAmount = float(request.form['loanamount'])
    Credit_History = float(request.form['credithistory'])
    holdername=request.form['holdernames']

    # Prepare input for model
    input_data = np.array([[ApplicantIncome, LoanAmount, Credit_History]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Interpret the result
    if prediction == 1:
        result = f"{holdername}Loan is likely to be approved."
    else:
        result = f"{holdername}Loan is likely to be rejected."

    # Return result to template
    return render_template('proj.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
